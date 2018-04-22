namespace NNClass
open System;
open System.IO;
open Akka.Configuration;
open System.Threading;
open System.Threading.Tasks;

type Param_Msg =
    | Get of int * int * int * AsyncReplyChannel<double[][] * double[] * double[][] * double[]>
    | Update of int * int * int * double[,] * double[] * double[,] * double[]
    | SaveModel of string * AsyncReplyChannel<unit>

type NeuralNetwork (numInput: int, numHidden: int, numOutput: int, seed: int) =
    let numWeights = 
        (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

    let rnd = new Random (seed); // used by randomize and by train/shuffle
    
    let inputs = Array.create numInput 0.0;
    let hOutputs = Array.create numHidden 0.0;
    let outputs = Array.create numOutput 0.0;

    let mutable ihWeights: double[][] = Array.init numInput (fun r -> Array.create numHidden 0.0); // input-hidden
    let mutable hBiases: double[] = Array.create numHidden 0.0;
    let mutable hoWeights: double[][] = Array.init numHidden (fun r -> Array.create numOutput 0.0) // hidden-output
    let mutable oBiases: double[] = Array.create numOutput 0.0;

    // back-prop momentum specific arrays
    let ihPrevWeightsDelta = Array2D.init numInput numHidden (fun i j -> 0.0);
    let hPrevBiasesDelta = Array.create numHidden 0.0;
    let hoPrevWeightsDelta = Array2D.init numHidden numOutput (fun i j -> 0.0);
    let oPrevBiasesDelta = Array.create numOutput 0.0; 

    let mutable minTrainErr = 1.0
    let mutable preTrainErr = 1.0

    member this.GetWeights (): double[] =
        let result = Array.create numWeights 0.0;
        let mutable k = 0;

        for i = 0 to ihWeights.Length-1 do
            for j = 0 to ihWeights.[i].Length-1 do
                result.[k] <- ihWeights.[i].[j];
                k <- k + 1
        for i = 0 to hBiases.Length-1 do
            result.[k] <- hBiases.[i];
            k <- k + 1
        for i = 0 to hoWeights.Length-1 do
            for j = 0 to hoWeights.[i].Length-1 do
                result.[k] <- hoWeights.[i].[j];
                k <- k + 1
        for i = 0 to oBiases.Length-1 do
            result.[k] <- oBiases.[i];
            k <- k + 1
        result;
        
    member this.SetWeights (weights: double[]) =
        // copy serialized weights and biases in weights[] array
        // to i-h weights, h biases, h-o weights, o biases
        if weights.Length <> numWeights then
            raise (new Exception ("Bad weights array in SetWeights"));

        let mutable k = 0; // points into weights param

        for i = 0 to numInput-1 do
            for j = 0 to numHidden-1 do
                ihWeights.[i].[j] <- weights.[k];
                k <- k + 1
        for i = 0 to numHidden-1 do
            hBiases.[i] <- weights.[k];
            k <- k + 1
        for i = 0 to numHidden-1 do
            for j = 0 to numOutput-1 do
                hoWeights.[i].[j] <- weights.[k];
                k <- k + 1
        for i = 0 to numOutput-1 do
            oBiases.[i] <- weights.[k];
            k <- k + 1
        
    member this.XavierWeights () = 
        // initialize weights and biases approx. Xavier
        let var fanIn fanOut = 2.0 / float (fanIn + fanOut)
        let gaussian mean stddev = 
            let u1 = 1.0 - rnd.NextDouble();
            let u2 = 1.0 - rnd.NextDouble();
            let randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            let randNormal = mean + stddev * randStdNormal;
            randNormal

        for i = 0 to numInput-1 do
            for j = 0 to numHidden-1 do
                let stddev = Math.Sqrt(var numInput numHidden);
                ihWeights.[i].[j] <- gaussian 0.0 stddev;
        for i = 0 to numHidden-1 do
            let stddev = Math.Sqrt(var numInput numOutput);
            hBiases.[i] <- gaussian 0.0 stddev;
        for i = 0 to numHidden-1 do
            for j = 0 to numOutput-1 do
                let stddev = Math.Sqrt(var numHidden numOutput);
                hoWeights.[i].[j] <- gaussian 0.0 stddev;
        for i = 0 to numOutput-1 do
            let stddev = Math.Sqrt(var numHidden 1);
            oBiases.[i] <- gaussian 0.0 stddev;
        
    member this.RandomiseWeights () = 
        // initialize weights and biases to small random values
        let initFun i = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001
        let initialWeights = Array.init numWeights (initFun);
        this.SetWeights (initialWeights);
        
    static member LoadModel (modelName: string, nnSeed:int) =
        let lines = 
            File.ReadAllLines (modelName)
            |> Array.map (fun line -> 
                let c = line.IndexOf ("//")
                if c >= 0 then line.Substring (0, c)
                else line)
            |> Array.filter (fun line -> line.Trim() <> "")
        
        let nums =
            lines.[0].Split(' ')
            |> Array.collect (fun tok -> 
                if tok.Trim() <> "" then [| Int32.Parse (tok) |] else [| |]) 
        
        let numInput, numHidden, numOutput = nums.[0], nums.[1], nums.[2]
        //printfn "nums: %A" (numInput, numHidden, numOutput)
        
        let weights = 
            lines.[1..]
            |> Array.collect (fun line -> 
                line.Split(' ')
                |> Array.collect (fun tok -> 
                    if tok.Trim() <> "" then [| Double.Parse (tok) |] else [| |]
                    )
            )
        //printfn "weights (%d): %A" weights.Length weights
        
        let nn = NeuralNetwork (numInput, numHidden, numOutput, nnSeed);
        nn.SetWeights (weights);
        nn
      
    member this.SaveModel (modelName: string) =
        use tw = File.CreateText (modelName);
        this.FPrintModel (tw)
        
    member this.FPrintModel (tw: TextWriter) =
        fprintfn tw "// numInput numHidden numOutput"
        fprintf tw "%d %d %d\n" numInput numHidden numOutput

        fprintfn tw "\n// i-h weights (%i*%i):" numInput numHidden
        for i = 0 to numInput-1 do
            ihWeights.[i] |> Array.iter (fun v -> fprintf tw "% .4f " v)
            fprintf tw "\n"

        fprintfn tw "\n// h biases (%i):" numHidden
        hBiases |> Array.iter (fun v -> fprintf tw "% .4f " v)
        fprintf tw "\n"
        
        fprintfn tw "\n// h-o weights (%i*%i):" numHidden numOutput
        for i = 0 to numHidden-1 do
            hoWeights.[i] |> Array.iter (fun v -> fprintf tw "% .4f " v)
            fprintf tw "\n"
        
        fprintfn tw "\n// o biases (%i):" numOutput
        oBiases |> Array.iter (fun v -> fprintf tw "% .4f " v)
        fprintf tw "\n"

        fprintfn tw "\n//"

    member this.HyperTanFunction (x: double): double =
        if x < -20.0 then -1.0; // approximation is correct to 30 decimals
        elif x > 20.0 then 1.0;
        else Math.Tanh (x);

    member this.Softmax (oSums: double[]): double[] =
        // determine max output sum
        // does all output nodes at once so scale doesn't have to be re-computed each time
        
        let sum = Array.sumBy (fun v -> Math.Exp(v)) oSums;
        let result = Array.map (fun v -> Math.Exp(v) / sum) oSums;
        result; // now scaled so that xi sum to 1.0

    member this.Shuffle (sequence: int[]) = // Fisher
        for i = 0 to sequence.Length-1 do
            let r = rnd.Next (i, sequence.Length);
            let tmp = sequence.[r];
            sequence.[r] <- sequence.[i];
            sequence.[i] <- tmp;

    member this.ComputeOutputs (xValues: double[]): double[] =
        let hSums = Array.create numHidden 0.0 // hidden nodes sums scratch array
        let oSums = Array.create numOutput 0.0; // output nodes sums

        for i in 0 .. xValues.Length-1 do // copy x-values to inputs
            inputs.[i] <- xValues.[i];
        // note: no need to copy x-values unless you implement a ToString.
        // more efficient is to simply use the xValues[] directly.

        for j in 0 .. numHidden-1 do  // compute sum of (ia) weights * inputs
            for i in 0 .. numInput-1 do
                hSums.[j] <- hSums.[j] + inputs.[i] * ihWeights.[i].[j]; 

        for i in 0 .. numHidden-1 do  // add biases to a sums
            hSums.[i] <- hSums.[i] + hBiases.[i];

        for i in 0 .. numHidden-1 do   // apply activation
            hOutputs.[i] <- this.HyperTanFunction (hSums.[i]); // hard-coded

        for j in 0 .. numOutput-1 do  // compute h-o sum of weights * hOutputs
            for i in 0 .. numHidden-1 do
                oSums.[j] <- oSums.[j] + hOutputs.[i] * hoWeights.[i].[j]; 

        for i in 0 .. numOutput-1 do  // add biases to input-to-hidden sums
            oSums.[i] <- oSums.[i] + oBiases.[i];

        let softOut = this.Softmax (oSums); // all outputs at once for efficiency
        Array.blit softOut 0 outputs 0 softOut.Length;

        let retResult = Array.copy outputs; // could define a GetOutputs method
        retResult;

    member this.Error (trainData: double[][]): double =
        // average squared error per training item
        let mutable sumSquaredError = 0.0;
        
        for i = 0 to trainData.Length - 1 do
            let xValues = Array.sub trainData.[i] 0 numInput; // inputs
            let tValues = Array.sub trainData.[i] numInput numOutput; // target values
            let yValues = this.ComputeOutputs xValues; // outputs using current weights

            sumSquaredError <- 
                sumSquaredError + (Array.zip tValues yValues
                                   |> Array.sumBy(fun (a, b) -> (a - b) * (a - b)));
        sumSquaredError / (float trainData.Length);

    member this.MaxIndex (vector : double[]): int = // helper for Accuracy()
        // index of largest value
        let mutable maxIdx = 0;
        let mutable maxVal = vector.[0];
        for i = 0 to vector.Length - 1 do
            if vector.[i] > maxVal then
                maxVal <- vector.[i];
                maxIdx <- i;
        maxIdx;

    member this.Accuracy (testData: double[][]): double =
        // percentage correct using winner-takes all
        let mutable numCorrect = 0.0;
        let mutable numWrong = 0.0;
        
        for i = 0 to testData.Length - 1 do
            let xValues = Array.sub testData.[i] 0 numInput; // get inputs
            let tValues = Array.sub testData.[i] numInput numOutput; // get target values
            let yValues = this.ComputeOutputs xValues; // outputs using current weights

            let maxIndex = this.MaxIndex yValues;
            let tMaxIndex = this.MaxIndex tValues;

            if maxIndex = tMaxIndex then
                numCorrect <- numCorrect + 1.0;
            else
                numWrong <- numWrong + 1.0;
        numCorrect / (numCorrect + numWrong);

    member this.Train (paramstoreStore: MailboxProcessor<Param_Msg>, 
                       datashardIdx: int, 
                       trainData: double[][], 
                       maxEpochs: int, 
                       learnRate: double, 
                       momentum: double, 
                       errtw: TextWriter,
                       acount: int ref,
                       adone: TaskCompletionSource<bool>) =
      try
        // train using back-prop
        
        // train a back-prop style NN classifier using learning rate and momentum
        let errInterval = maxEpochs / 10; // interval to check validation data
        let sequence = [|0 .. trainData.Length - 1|]

        let datashardActor = MailboxProcessor<bool>.Start(fun inbox ->
            let rec loop epoch = async {
                // log trainning error
                let trainErr = this.Error trainData;
                let c = 
                    if minTrainErr > trainErr then "*"
                    else if preTrainErr > trainErr then "-"
                    else ""
                // TODO
                // fprintf errtw "%4i %.4f %s\n" epoch trainErr c
                minTrainErr <- min trainErr minTrainErr
                preTrainErr <- trainErr

                if (epoch % errInterval) = 0 && epoch <= maxEpochs then
                    printfn "index = %i  epoch = %4i  training error = %.4f" datashardIdx epoch trainErr;

                if epoch >= maxEpochs then
                    if Interlocked.Decrement acount = 0 then 
                        //Console.WriteLine (sprintf "... [%d] done all" (tid()))
                        adone.SetResult true
                    return ()
            
                else
                    this.Shuffle(sequence) // visit each training data in random order

                    for ii = 0 to trainData.Length - 1 do

                        let _ihWeights, _hBiases, _hoWeights, _oBiases = paramstoreStore.PostAndReply (fun ch -> Get (datashardIdx, epoch, ii, ch))
                        ihWeights <- Array.copy _ihWeights
                        hBiases <- Array.copy _hBiases
                        hoWeights <- Array.copy _hoWeights
                        oBiases <- Array.copy _oBiases 

                        let idx = sequence.[ii]
                        let xValues = Array.sub trainData.[idx] 0 numInput; // inputs
                        let tValues = Array.sub trainData.[idx] numInput numOutput; // target values
                        this.ComputeOutputs xValues |> ignore; // copy xValues in, compute outputs

                        // 1. compute output nodes signals (assumes softmax)
                        // output signals - gradients w/o associated input terms
                        let oSignals = 
                            Array.init numOutput (fun k -> 
                                (tValues.[k] - outputs.[k]) * (1.0 - outputs.[k]) * outputs.[k]);
                
                        // 2. compute hidden-to-output weights gradients using output signals
                        let hoGrads = 
                            Array2D.init numHidden numOutput (fun j k -> oSignals.[k] * hOutputs.[j]);

                        // 2b. compute output biases gradients using output signals
                        // TODO unnecessary convertion
                        let obGrads = 
                            Array.init numOutput (fun k -> oSignals.[k] * 1.0); // dummy assoc. input values

                        // 3. compute hidden nodes signals
                        let hSignals: double[] = 
                            Array.init numHidden (fun j -> 
                                let sum = 
                                    Array.zip oSignals hoWeights.[j]
                                    |> Array.sumBy(fun (a, b) -> a * b);
                                (1.0 + hOutputs.[j]) * (1.0 - hOutputs.[j]) * sum);

                        // 4. compute input0hidden weights gradients
                        let ihGrads = Array2D.init numInput numHidden (fun i j -> hSignals.[j] * inputs.[i]);

                        // 4b. compute hidden node biases gradients
                        let hbGrads = Array.init numHidden (fun j -> hSignals.[j] * 1.0); // dummy 1.0 input

                        paramstoreStore.Post (Update (datashardIdx, epoch, ii, ihGrads, hbGrads, hoGrads, obGrads))

                        return! loop (epoch + 1)
                }
            loop 0)
        ()
      with ex -> 
        Console.WriteLine ("*** {0}", ex.Message);
        ()

    member this.GetCurrentWeights () = 
        (ihWeights, hBiases, hoWeights, oBiases)

    member this.UpdateWeights (ihGrads: double[,])
                              (hbGrads: double[])
                              (hoGrads: double[,])
                              (obGrads: double[])
                              (learnRate: double) 
                              (momentum: double) = 
        try
            // update input-to-hidden weights
            for i = 0 to numInput - 1 do
                for j = 0 to numHidden - 1 do
                    let delta = ihGrads.[i,j] * learnRate;
                    ihWeights.[i].[j] <- ihWeights.[i].[j] + delta;
                    ihWeights.[i].[j] <- ihWeights.[i].[j] + ihPrevWeightsDelta.[i,j] * momentum;
                    ihPrevWeightsDelta.[i,j] <- delta // save for next time
                
            // update hidden biases
            for j = 0 to numHidden - 1 do
                let delta = hbGrads.[j] * learnRate;
                hBiases.[j] <- hBiases.[j] + delta;
                hBiases.[j] <- hBiases.[j] + hPrevBiasesDelta.[j] * momentum;
                hPrevBiasesDelta.[j] <- delta;

            // update hiiden-to-output weights
            for j = 0 to numHidden - 1 do
                for k = 0 to numOutput - 1 do
                    let delta = hoGrads.[j,k] * learnRate;
                    hoWeights.[j].[k] <- hoWeights.[j].[k] + delta;
                    hoWeights.[j].[k] <- hoWeights.[j].[k] + hoPrevWeightsDelta.[j,k] * momentum;
                    hoPrevWeightsDelta.[j,k] <- delta;

            // update output node biases
            for k = 0 to numOutput - 1 do
                let delta = obGrads.[k] * learnRate;
                oBiases.[k] <- oBiases.[k] + delta;
                oBiases.[k] <- oBiases.[k] + oPrevBiasesDelta.[k] * momentum;
                oPrevBiasesDelta.[k] <- delta;
            ()
        with ex -> 
            Console.WriteLine ("Error during weight updates {0}", ex.Message);

// F# |> I LOVE