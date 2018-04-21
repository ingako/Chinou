module NNModel
open System;
open System.IO;
open Akka.Configuration;
open NNClass
open NNHelper
        
let Main2 (hoconName: string) (job: string) (x: string) =
    printfn "... hoconName = %s" hoconName
    printfn "... job = %s" job
    printfn "... x = %s" x
    
    let job = job.ToUpper ()
    if (job <> "/TRAIN" && job <> "/TEST") then
        failwith "Job name must be either /TRAIN or /TEST"
    
    let x = x.ToUpper ()
    if (not (String.IsNullOrEmpty x) && x <> "/X+" && x <> "/X-") then
        failwith "If given, x must be either /X+ or /X-"
    let xavier = Some (x = "/X+")
    
    let hocon = File.ReadAllText (hoconName);
    printfn ""
    //printfn "... %s\n" hocon
    let config = ConfigurationFactory.ParseString (hocon);

    let taskName = config.GetString ("root.taskName");
    let dataName = taskName + "-data.txt"
    let data2Name = taskName + "-data2.txt"
    let errName = taskName + "-err.log"
    let modelName = taskName + "-model.txt"
    printfn "... taskName = %s" taskName
    printfn "... dataName = %s" dataName
    printfn "... data2Name = %s" data2Name
    printfn "... errName = %s" errName
    printfn "... modelName = %s" modelName

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
    
    if job = "/TRAIN" then
        printfn "\n--- Training a model ---"
        
        printfn "\nLoading data from %s" dataName
        let allData = NNHelper.LoadData (dataName)
        printfn "The %d-item data set is:\n" allData.Length
        NNHelper.ShowMatrix (allData, numRows=4, decimals=1, indices=true)

        printfn "\nSplitting data into %.0f%% train, %.0f%% test" (trainPct*100.0) (100.0-trainPct*100.0)
        let trainData, testData = NNHelper.SplitData (allData, trainPct, splitSeed);
        
        printfn "\nThe training data is:\n"    
        NNHelper.ShowMatrix (trainData, numRows=4, decimals=1, indices=true)
        printfn "\nThe test data is:\n"     
        NNHelper.ShowMatrix (testData, numRows=3, decimals=1, indices=true);

        printfn "\nCreating a %d-%d-%d neural network" numInput numHidden numOutput
        let nntrain = NeuralNetwork (numInput, numHidden, numOutput, nnSeed)

        printfn "\nRandomise weights with seed %d" nnSeed
        if xavier then nntrain.XavierWeights ()
        else nntrain.RandomiseWeights ()
        
        printfn "\nThe initial weights and biases are:"
        let weights = nntrain.GetWeights ()
        NNHelper.ShowVector (weights, decimals=4, lineLen=10);

        let allAcc = nntrain.Accuracy (allData)
        let trainAcc = nntrain.Accuracy (trainData)
        let testAcc = nntrain.Accuracy (testData)

        printfn "\nAccuracy on all data   = %.4f" allAcc
        printfn "Accuracy on train data = %.4f" trainAcc
        printfn "Accuracy on test data  = %.4f" testAcc

        printfn "\nSetting maxEpochs = %d" maxEpochs
        printfn "Setting learnRate = %.4f" learnRate
        printfn "Setting momentum  = %.4f" momentum

        use errtw = File.CreateText (errName);

        printfn "\nStarting training\n"
        let weights = nntrain.Train (trainData, maxEpochs, learnRate, momentum, errtw)
        printfn "\nTraining complete"

        printfn "\nThe trained weights and biases are:"
        NNHelper.ShowVector (weights, decimals=4, lineLen=10);

        let allAcc = nntrain.Accuracy (allData)
        let trainAcc = nntrain.Accuracy (trainData)
        let testAcc = nntrain.Accuracy (testData)

        printfn "\nAccuracy on all data   = %.4f" allAcc
        printfn "Accuracy on train data = %.4f" trainAcc
        printfn "Accuracy on test data  = %.4f" testAcc

        printfn "\nSaving model as %s" modelName
        nntrain.SaveModel (modelName)

    printfn "\n--- Testing a saved model ---"
    
    printfn "\nLoading data from %s" data2Name
    let testData2 = NNHelper.LoadData (data2Name)
    printfn "The %d-item data set is:\n" testData2.Length
    NNHelper.ShowMatrix (testData2, numRows=4, decimals=1, indices=true)
    
    printfn "\nLoading model from %s" modelName
    let nnload = NeuralNetwork.LoadModel (modelName, nnSeed)

    printfn "\nThe loaded weights and biases are:"
    let weights = nnload.GetWeights ()
    NNHelper.ShowVector (weights, decimals=4, lineLen=10);

    let testAcc = nnload.Accuracy (testData2)

    printfn "\nAccuracy on test data 2 = %.4f" testAcc

    printfn "\nEnd neural network model demo\n"

[<EntryPoint>]
let Main (args: string[]) =
    try 
        // args.[0] HOCON config 
        // args.[1] a tag (/TRAIN or /TEST)
        // args.[2] use xavier initialization or not (/X+ or /X-)
        Main2 args.[0] args.[1] (if args.Length > 2 then args.[2] else "")
        Console.ReadLine() |> ignore;
        0;

    with ex ->
        printfn "*** Exception: %s" ex.Message
        eprintfn "*** Exception: %s" ex.Message
        1
