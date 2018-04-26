module NNModel
open System;
open System.IO;
open Akka.Configuration;
open System.Threading;
open System.Threading.Tasks;
open NNClass
open NNHelper
        
let Main2 (hoconName: string) (job: string) (nShards: string) (x: string) =
    printfn "... hoconName = %s" hoconName
    printfn "... job = %s" job
    printfn "... n = %s" nShards
    printfn "... x = %s" x
    
    let job = job.ToUpper ()
    if (job <> "/TRAIN" && job <> "/TEST") then
        failwith "Job name must be either /TRAIN or /TEST"
    
    let x = x.ToUpper ()
    if (not (String.IsNullOrEmpty x) && x <> "/X+" && x <> "/X-") then
        failwith "If given, x must be either /X+ or /X-"
    let xavier = Some (x = "/X+")
    
    // number of data shards
    let N = int (nShards.Split ':').[1]

    let hocon = File.ReadAllText (hoconName);
    printfn ""
    //printfn "... %s\n" hocon
    let config = ConfigurationFactory.ParseString (hocon);

    let taskName = config.GetString ("root.taskName");
    let dataName = taskName + "-data"
    let data2Name = taskName + "-data2.txt"
    
    let modelName = taskName + "-model.txt"
    printfn "... taskName = %s" taskName
    
    let dataName = [| for idx = 0 to N - 1 do yield sprintf "%s-data-%i.txt" taskName idx |]
    let errName = [| for idx = 0 to N - 1 do yield sprintf "%s-err-%i.log" taskName idx |]
    for idx = 0 to N - 1 do
        printfn "... dataName.[%i] = %s" idx dataName.[idx]
        printfn "... errName.[%i] = %s" idx errName.[idx]
    
    printfn "... modelName = %s" modelName
    printfn "... data2Name = %s" data2Name

    let trainPct = config.GetDouble ("root.trainPct");
    printfn "... trainPct = %f" trainPct
    let splitSeed = config.GetInt ("root.splitSeed");
    printfn "... splitSeed = %d" splitSeed
    let numInput = config.GetInt ("root.numInput");
    printfn "... numInput = %d" numInput
    let numHidden = config.GetInt ("root.numHidden");
    printfn "... numHidden = %d" numHidden
    let numOutput = config.GetInt ("root.numOutput");
    printfn "... numOutput = %d" numOutput
    let nnSeed = config.GetInt ("root.nnSeed");
    printfn "... nnSeed = %d" nnSeed
    
    let xavier = 
        if Option.isSome xavier then Option.get xavier 
        else config.GetBoolean ("root.xavier");
    printfn "... xavier = %b" xavier
    
    let maxEpochs = config.GetInt ("root.maxEpochs");
    printfn "... maxEpochs = %d" maxEpochs
    let learnRate = config.GetDouble ("root.learnRate");
    printfn "... learnRate = %f" learnRate
    let momentum = config.GetDouble ("root.momentum");
    printfn "... momentum = %f" momentum
    
    // random number generator to be used for all the neural network instances
    let rnd = new Random (nnSeed)

    if job = "/TRAIN" then
        printfn "\n--- Training a distributed model ---"

        printfn "\n--- Parameter shard ---"
        // set up and kick off parameter store actor
        let paramstore (nntrain: NeuralNetwork) (learnRate: double) (momentum: double) =
            new MailboxProcessor<Param_Msg> (fun inbox ->
                let rec loop () = async {
                    let! m = inbox.Receive ()
                    match m with
                    | Update (index, epoch, i, ihGrads, hbGrads, hoGrads, obGrads) ->
                        nntrain.UpdateWeights ihGrads hbGrads hoGrads obGrads learnRate momentum
                        return! loop ()
                    | Get (index, epoch, i, ch) ->
                        // reply on the given ch with the nntrain weights and biases
                        // printfn "epoch = %i dataIdx = %i" epoch i
                        ch.Reply (nntrain.GetCurrentWeights ())
                        return! loop ()
                    | SaveModel (modelName, ch) ->
                        printfn "\nSaving model as %s" modelName
                        nntrain.SaveModel modelName
                        ch.Reply()
                        ()
                    }
                loop ())
        let nnParamStore = NeuralNetwork (numInput, numHidden, numOutput, rnd)

        printfn "\nRandomise weights with seed %d" nnSeed
        if xavier then nnParamStore.XavierWeights ()
        else nnParamStore.RandomiseWeights ()

        printfn "\nThe initial weights and biases are:"
        let weights = nnParamStore.GetWeights ()
        NNHelper.ShowVector (weights, decimals=4, lineLen=10)

        let paramstoreStore = paramstore nnParamStore learnRate momentum
        paramstoreStore.Start ()
        
        // set up termination helpers
        let acount = ref N
        let adone = TaskCompletionSource<bool> ()

        // interval to check validation data
        let errInterval = maxEpochs / 10 

        // kick off datashard actors
        let datashardActors = [|
            for datashardIdx = 0 to N - 1 do
                printfn "\n--- Data shard %i ---" datashardIdx
                let dataShardName = dataName.[datashardIdx]
                printfn "\nLoading data from %s" dataShardName
                let allData = NNHelper.LoadData (dataShardName)
                printfn "The %d-item data set is:\n" allData.Length
                NNHelper.ShowMatrix (allData, numRows=4, decimals=1, indices=true)

                printfn "\nSplitting data into %.0f%% train, %.0f%% test" (trainPct*100.0) (100.0-trainPct*100.0)
                let trainData, testData = NNHelper.SplitData (allData, trainPct, splitSeed);
                
                printfn "\nThe training data is:\n"    
                NNHelper.ShowMatrix (trainData, numRows=4, decimals=1, indices=true)
                
                if trainData.Length = 0 then
                    // handle empty training data
                    decr acount

                else 
                    printfn "\nCreating a %d-%d-%d neural network" numInput numHidden numOutput
                    let nntrain = NeuralNetwork (numInput, numHidden, numOutput, rnd)

                    let errtw = File.CreateText (errName.[datashardIdx])
                    
                    yield nntrain.Train (paramstoreStore, datashardIdx, trainData, maxEpochs, learnRate, momentum, errtw, acount, adone)      
            |]

        // kick off datashard actors
        printfn "\nStarting training\n"
        datashardActors |> Array.iter(fun a -> a.Start())

        adone.Task.Wait ()
        printfn "\nTraining complete"

        // write model to file
        paramstoreStore.PostAndReply (fun ch -> SaveModel (modelName, ch))

    printfn "\n--- Testing a saved model ---"
    
    printfn "\nLoading data from %s" data2Name
    let testData2 = NNHelper.LoadData (data2Name)
    printfn "The %d-item data set is:\n" testData2.Length
    NNHelper.ShowMatrix (testData2, numRows=4, decimals=1, indices=true)
    
    printfn "\nLoading model from %s" modelName
    let nnload = NeuralNetwork.LoadModel (modelName, rnd)

    printfn "\nThe loaded weights and biases are:"
    let weights = nnload.GetWeights ()
    NNHelper.ShowVector (weights, decimals=4, lineLen=10)

    let testAcc = nnload.Accuracy (testData2)

    printfn "\nAccuracy on test data 2 = %.4f" testAcc

    printfn "\nEnd neural network model demo\n"

[<EntryPoint>]
let Main (args: string[]) =
    try 
        // args.[0] HOCON config 
        // args.[1] a tag (/TRAIN or /TEST)
        // args.[2] number of data shards
        // arsg.[3] use xavier initialization or not (/X+ or /X-)
        Main2 args.[0] args.[1] args.[2] (if args.Length > 3 then args.[3] else "")
        Console.ReadLine() |> ignore
        0;

    with ex ->
        printfn "Something horrible happened: %s" ex.Message
        eprintfn "Something horrible happened: %s" ex.Message
        1
