namespace NNHelper
open System;
open System.IO;

type NNHelper = 
    static member ShowVector (vector: double[], decimals: int, lineLen: int) =
        let formatv = Printf.TextWriterFormat<float->unit>(sprintf "%%%d.%df " (decimals+3) decimals)
        
        vector |> Array.iteri (fun i v ->
            if i > 0 && i % lineLen = 0 then printfn "";
            printf formatv v
            )
            
        //for i in 0 .. vector.Length-1 do
            //if i > 0 && i % lineLen = 0 then printfn "";
            //printf formatv vector.[i]
        
        printfn "";

    static member ShowMatrix (matrix: double[][], numRows: int, decimals: int, indices: bool) =
        let formatv = Printf.TextWriterFormat<float->unit>(sprintf "%%%d.%df  " (decimals+3) decimals)
        
        let len = matrix.Length.ToString().Length;
        let formati = Printf.TextWriterFormat<int->unit>(sprintf "[%%%dd]  " len)
        
        matrix 
            |> Array.take (min numRows matrix.Length) 
            |> Array.iteri (fun i row ->
                if indices then printf formati i 
                row |> Array.iter (fun v -> printf formatv v)
                printfn "" )

        if numRows < matrix.Length then
            printfn ". . ."
            let lastRow = matrix.Length - 1
            if indices then printf formati lastRow
            matrix.[lastRow] |> Array.iter (fun v -> printf formatv v)

        printfn ""

    static member LoadData (dataFile: string) : double[][] =
        let lines = File.ReadAllLines (dataFile);
        let result =
            lines
            |> Array.map (fun line -> 
                let c = line.IndexOf ("//")
                if c >= 0 then line.Substring (0, c)
                else line)
            |> Array.filter (fun line -> line.Trim() <> "")
            |> Array.map (fun line -> 
                line.Split(' ') 
                |> Array.collect (fun tok -> 
                    if tok.Trim() <> "" then [| Double.Parse (tok) |] else [| |]))
        result
            
    static member SplitData (allData: double[][], trainPct: double, splitSeed: int): (double[][] * double[][]) = // trainData, testData
        let rnd = new Random (splitSeed);
        let totRows = allData.Length;
        let numTrainRows = (int)((double)totRows * trainPct); // usually 0.80
        let numTestRows = totRows - numTrainRows;
        //let trainData = Array.create numTrainRows Array.create 0 0.0;
        //let testData = Array.create numTestRows Array.create 0 0.0;

        let copy = Array.copy allData

        /// call this.Shuffle ?
        for i in 0 .. copy.Length-1 do // scramble order
            let r = rnd.Next (i, copy.Length); // use Fisher-Yates
            let tmp = copy.[r];
            copy.[r] <- copy.[i];
            copy.[i] <- tmp;

        let trainData = copy.[0..numTrainRows-1]; // numTrainRows
        let testData = copy.[numTrainRows..];     // numTestRows
        trainData, testData

