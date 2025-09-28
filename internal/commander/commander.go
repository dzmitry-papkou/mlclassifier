package commander

import (
    "bufio"
    "encoding/csv"
    "fmt"
    "math"
    "math/rand"
    "os"
    "path/filepath"
    "strconv"
    "strings"
    "time"

    "mlclassifier/internal/evaluation"
    "mlclassifier/internal/jobs"
    "mlclassifier/internal/models"
    "mlclassifier/internal/persistence"
    "mlclassifier/internal/preprocessing"

    "github.com/fatih/color"
    "github.com/shopspring/decimal"
)

type Commander struct {
    currentModel      models.Model
    modelBundle       *persistence.ModelBundle
    currentModelPath  string
    loadedData        *DataSet
    jobManager        *jobs.Manager

    green  func(a ...any) string
    red    func(a ...any) string
    yellow func(a ...any) string
    cyan   func(a ...any) string
    blue   func(a ...any) string
}

type DataSet struct {
    X           [][]decimal.Decimal
    y           []int
    Features    []string
    Classes     []string
    SourceFile  string
}

func NewCommander() *Commander {
    return &Commander{
        jobManager: jobs.NewManager(),
        green:      color.New(color.FgGreen).SprintFunc(),
        red:        color.New(color.FgRed).SprintFunc(),
        yellow:     color.New(color.FgYellow).SprintFunc(),
        cyan:       color.New(color.FgCyan).SprintFunc(),
        blue:       color.New(color.FgBlue).SprintFunc(),
    }
}

func (c *Commander) saveTrainingLog(modelName string, metadata persistence.BundleMetadata) {
    logFile := filepath.Join("models", "training_log.csv")
    file, err := os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
    if err != nil {
        return
    }
    defer file.Close()
    
    info, _ := file.Stat()
    if info.Size() == 0 {
        fmt.Fprintln(file, "Timestamp,Model,Dataset,Accuracy,TrainingTime")
    }
    
    fmt.Fprintf(file, "%s,%s,%s,%.4f,%.2f\n",
        time.Now().Format("2006-01-02 15:04:05"),
        modelName,
        metadata.Dataset,
        metadata.Accuracy,
        metadata.TrainingTime.Seconds())
}

func (c *Commander) Start() {
    c.printWelcome()
    scanner := bufio.NewScanner(os.Stdin)

    for {
        fmt.Print(c.yellow("\nmlc> "))
        if !scanner.Scan() {
            if scanner.Err() != nil {
                fmt.Printf("\n%s Scanner error: %v\n", c.red("✗"), scanner.Err())
            }
            break
        }

        input := strings.TrimSpace(scanner.Text())

        if input == "" {
            continue
        }

        parts := strings.Fields(input)
        command := strings.ToLower(parts[0])
        args := parts[1:]

        c.ExecuteCommand(command, args)
    }
}

func (c *Commander) ExecuteCommand(command string, args []string) {
    switch command {
    case "help", "h":
        c.showHelp()
    case "load":
        if len(args) > 0 {
            c.loadData(args[0])
        } else {
            fmt.Println(c.red("Usage: load <filename>"))
        }
    case "train":
        if len(args) > 0 {
            c.trainModel(args[0], args[1:])
        } else {
            c.showTrainHelp()
        }
    case "predict":
        c.predict(args)
    case "test":
        c.interactiveEvaluate()
    case "batch":
        if len(args) > 0 {
            c.batchPredict(args[0])
        } else {
            fmt.Println(c.red("Usage: batch <filename>"))
        }
    case "evaluate":
        c.evaluate()
    case "cv":
        c.crossValidate()
    case "loadmodel":
        if len(args) > 0 {
            c.loadModel(args[0])
        } else {
            fmt.Println(c.red("Usage: loadmodel <filename>"))
        }
    case "list":
        c.listModels()
    case "current":
        c.showCurrentModel()
    case "compare":
        c.compareModels()
    case "best":
        c.findBestModel()
    case "save-results":
        cvFolds := 5
        if len(args) > 0 {
            if folds, err := strconv.Atoi(args[0]); err == nil && folds > 1 {
                cvFolds = folds
            }
        }
        c.saveCurrentModelResults(cvFolds)
    case "info":
        c.showDataInfo()
    case "experiment":
        c.runExperiment()
    case "load-streaming":
        if len(args) > 0 {
            c.loadStreamingData(args)
        } else {
            fmt.Println(c.red("Usage: load-streaming <filename> --batch-size=1000"))
        }
    case "train-bg":
        if len(args) > 0 {
            c.trainModelBackground(args[0], args[1:])
        } else {
            fmt.Println(c.red("Usage: train-bg <algorithm> [params]"))
        }
    case "cv-advanced":
        c.crossValidateAdvanced()
    case "experiment-bg":
        c.runExperimentBackground()
    case "job-status":
        if len(args) > 0 {
            c.showJobStatus(args[0])
        } else {
            c.listAllJobs()
        }
    case "job-cancel":
        if len(args) > 0 {
            c.cancelJob(args[0])
        } else {
            fmt.Println(c.red("Usage: job-cancel <job-id>"))
        }
    case "job-logs":
        if len(args) > 0 {
            c.showJobLogs(args[0])
        } else {
            fmt.Println(c.red("Usage: job-logs <job-id>"))
        }
    case "model-versions":
        c.listModelVersions()
    case "model-compare":
        if len(args) >= 2 {
            c.compareModelVersions(args[0], args[1])
        } else {
            fmt.Println(c.red("Usage: model-compare <version1> <version2>"))
        }
    case "model-promote":
        if len(args) > 0 {
            c.promoteModel(args[0])
        } else {
            fmt.Println(c.red("Usage: model-promote <version>"))
        }
    case "model-rollback":
        if len(args) > 0 {
            c.rollbackModel(args[0])
        } else {
            fmt.Println(c.red("Usage: model-rollback <version>"))
        }
    case "experiments":
        c.listExperiments()
    case "view":
        if len(args) > 0 {
            c.viewExperiment(args[0])
        } else {
            fmt.Println(c.red("Usage: view <experiment-id>"))
        }
    case "clear":
        c.clearScreen()
    case "quit", "exit", "q":
        c.quit()
    default:
        fmt.Printf("%s Unknown command: %s\n", c.red("✗"), command)
        fmt.Println("Type 'help' for available commands")
    }
}

func (c *Commander) printWelcome() {
    fmt.Println(c.cyan("╔══════════════════════════════════════════╗"))
    fmt.Println(c.cyan("║       ML Classifier Commander v1.0        ║"))
    fmt.Println(c.cyan("║         Interactive ML Training           ║"))
    fmt.Println(c.cyan("╚══════════════════════════════════════════╝"))
    fmt.Println()
    fmt.Println("Type 'help' for available commands")
}

func (c *Commander) showHelp() {
    fmt.Println(c.blue("\nAvailable Commands:"))

    fmt.Println("\n" + c.cyan("Data Management:"))
    fmt.Println("  load <file>            - Load dataset from CSV")
    fmt.Println("  load-streaming <file>  - Load large dataset in batches")
    fmt.Println("  info                   - Show loaded data information")

    fmt.Println("\n" + c.cyan("Model Training:"))
    fmt.Println("  train <algo>           - Train a model (knn, tree, forest, bayes)")
    fmt.Println("  train-bg <algo>        - Train model in background")
    fmt.Println("                           Models are auto-saved to models/ directory")
    fmt.Println("  evaluate               - Evaluate current model")
    fmt.Println("  cv                     - Run cross-validation on current model")
    fmt.Println("  cv-advanced            - Advanced CV with multiple algorithms")
    fmt.Println("  save-results [folds]   - Save current model results to CSV (default: 5 folds)")

    fmt.Println("\n" + c.cyan("Model Management:"))
    fmt.Println("  list                   - List all saved models")
    fmt.Println("  loadmodel <file>       - Load a saved model")
    fmt.Println("  current                - Show current active model info")
    fmt.Println("  compare                - Compare multiple models")
    fmt.Println("  best                   - Find best performing model")

    fmt.Println("\n" + c.cyan("Model Versioning:"))
    fmt.Println("  model-versions         - List all model versions")
    fmt.Println("  model-compare <v1> <v2>- Compare two model versions")
    fmt.Println("  model-promote <ver>    - Promote model to production")
    fmt.Println("  model-rollback <ver>   - Rollback to previous version")

    fmt.Println("\n" + c.cyan("Predictions:"))
    fmt.Println("  predict                - Make predictions with current model")
    fmt.Println("  test                   - Interactive testing mode")
    fmt.Println("  batch <file>           - Batch predictions from CSV")

    fmt.Println("\n" + c.cyan("Experiments:"))
    fmt.Println("  experiment             - Run full experiment suite")
    fmt.Println("  experiment-bg          - Run experiment in background")
    fmt.Println("  experiments            - List past experiments")
    fmt.Println("  view <exp-id>          - View experiment details")

    fmt.Println("\n" + c.cyan("Job Management:"))
    fmt.Println("  job-status [job-id]    - Show job status or list all jobs")
    fmt.Println("  job-cancel <job-id>    - Cancel a running job")
    fmt.Println("  job-logs <job-id>      - View job logs")

    fmt.Println("\n" + c.cyan("System:"))
    fmt.Println("  help                   - Show this help message")
    fmt.Println("  clear                  - Clear screen")
    fmt.Println("  quit                   - Exit program")
}

func (c *Commander) loadData(filename string) {
    startTime := time.Now()
    fmt.Printf("Loading data from %s...\n", filename)
    
    file, err := os.Open(filename)
    if err != nil {
        fmt.Printf("%s Error: %v\n", c.red("✗"), err)
        return
    }
    defer file.Close()
    
    fileInfo, _ := file.Stat()
    fileSize := fileInfo.Size()
    
    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        fmt.Printf("%s Error reading CSV: %v\n", c.red("✗"), err)
        return
    }
    
    if len(records) < 2 {
        fmt.Printf("%s No data found\n", c.red("✗"))
        return
    }
    
    headers := records[0]
    data := records[1:]
    
    missingCount := 0
    
    X := make([][]decimal.Decimal, 0, len(data))
    labels := make([]string, 0, len(data))
    
    for _, record := range data {
        hasEmpty := false
        for _, val := range record {
            if strings.TrimSpace(val) == "" {
                hasEmpty = true
                missingCount++
                break
            }
        }
        
        if hasEmpty {
            continue
        }
        
        row := make([]decimal.Decimal, len(record)-1)
        for j := 0; j < len(record)-1; j++ {
            val, err := decimal.NewFromString(record[j])
            if err != nil {
                fmt.Printf("%s Warning: Non-numeric value at row %d, col %d: %s\n", 
                    c.yellow("⚠"), len(X)+1, j+1, record[j])
                val = decimal.Zero
            }
            row[j] = val
        }
        X = append(X, row)
        labels = append(labels, record[len(record)-1])
    }
    
    encoder := preprocessing.NewLabelEncoder()
    y, _ := encoder.FitTransform(labels)
    
    classCount := make(map[int]int)
    for _, label := range y {
        classCount[label]++
    }
    
    minCount := len(y)
    maxCount := 0
    for _, count := range classCount {
        if count < minCount {
            minCount = count
        }
        if count > maxCount {
            maxCount = count
        }
    }
    
    imbalanceRatio := float64(maxCount) / float64(minCount)
    
    c.loadedData = &DataSet{
        X:          X,
        y:          y,
        Features:   headers[:len(headers)-1],
        Classes:    c.mapToSlice(encoder.IntToClass),
        SourceFile: filename,
    }
    
    loadTime := time.Since(startTime)
    
    fmt.Printf("%s Data loaded successfully!\n", c.green("✓"))
    fmt.Println(strings.Repeat("─", 50))
    fmt.Printf("File size:     %.2f KB\n", float64(fileSize)/1024)
    fmt.Printf("Load time:     %.3fs\n", loadTime.Seconds())
    fmt.Printf("Samples:       %d (skipped %d with missing values)\n", len(X), missingCount)
    fmt.Printf("Features:      %d\n", len(X[0]))
    fmt.Printf("Classes:       %d\n", len(encoder.IntToClass))
    
    fmt.Print("Distribution:  ")
    for class, name := range encoder.IntToClass {
        fmt.Printf("%s:%d ", name, classCount[class])
    }
    fmt.Println()
    
    if imbalanceRatio > 2 {
        fmt.Printf("%s Class imbalance detected (ratio: %.2f)\n", 
            c.yellow("⚠"), imbalanceRatio)
        fmt.Println("  Consider using stratified splitting or balancing techniques")
    }
    
    if missingCount > 0 {
        fmt.Printf("%s Dropped %d rows with missing values\n", 
            c.yellow("⚠"), missingCount)
    }
    
    fmt.Println(strings.Repeat("─", 50))
    fmt.Println("Ready to train! Use 'train <algorithm>' command")
}

func (c *Commander) trainModel(algorithm string, params []string) {
    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded. Use 'load <file>' first"))
        return
    }
    
    splitRatio := 0.8
    prepMethod := "raw"
    verbose := false
    
    for i := 0; i < len(params); i++ {
        if value, ok := strings.CutPrefix(params[i], "--split="); ok {
            val, _ := strconv.ParseFloat(value, 64)
            if val > 0 && val < 1 {
                splitRatio = val
            }
            params = append(params[:i], params[i+1:]...)
            i--
        } else if value, ok := strings.CutPrefix(params[i], "--prep="); ok {
            prepMethod = value
            params = append(params[:i], params[i+1:]...)
            i--
        } else if params[i] == "--verbose" {
            verbose = true
            params = append(params[:i], params[i+1:]...)
            i--
        }
    }
    
    fmt.Printf("Training %s model...\n", algorithm)
    fmt.Printf("Configuration: split=%.0f/%.0f, preprocessing=%s\n", 
        splitRatio*100, (1-splitRatio)*100, prepMethod)
    
    interrupt := make(chan bool, 1)
    done := make(chan bool, 1)

    stopProgress := make(chan bool, 1)
    go func() {
        if verbose {
            <-done
            return
        }

        spinner := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
        i := 0
        ticker := time.NewTicker(100 * time.Millisecond)
        defer ticker.Stop()

        for {
            select {
            case <-stopProgress:
                fmt.Print("\r")
                return
            case <-ticker.C:
                fmt.Printf("\r%s Training... ", spinner[i%len(spinner)])
                i++
            }
        }
    }()
    
    fmt.Println(c.yellow("Press 'stop' or 'q' to interrupt training..."))
    
    startTime := time.Now()
    
    var model models.Model
    var trainErr error
    
    go func() {
        defer func() {
            done <- true
        }()

        config := models.DefaultConfig(algorithm)

        switch algorithm {
        case "knn":
            if len(params) > 0 {
                config.K, _ = strconv.Atoi(params[0])
            }
            if len(params) > 1 {
                config.Distance = params[1]
            }

        case "tree":
            if len(params) > 0 {
                config.MaxDepth, _ = strconv.Atoi(params[0])
            }
            if len(params) > 1 {
                config.MinSplit, _ = strconv.Atoi(params[1])
            }

        case "forest":
            if len(params) > 0 {
                config.NTrees, _ = strconv.Atoi(params[0])
            }
            if len(params) > 1 {
                config.MaxDepth, _ = strconv.Atoi(params[1])
            }

        case "bayes":
            if len(params) > 0 {
                config.VarSmoothing, _ = strconv.ParseFloat(params[0], 64)
            }

        default:
            trainErr = fmt.Errorf("unknown algorithm: %s", algorithm)
            return
        }

        model, trainErr = models.CreateModel(config)
        if trainErr != nil {
            return
        }
        
        XProcessed := c.loadedData.X
        var scaler *preprocessing.Scaler

        switch prepMethod {
        case "normalized", "minmax":
            scaler = preprocessing.NewScaler("minmax")
            result, err := scaler.FitTransform(c.loadedData.X)
            if err != nil {
                XProcessed = c.loadedData.X
            } else {
                XProcessed = result
            }

        case "standardized", "standard":
            scaler = preprocessing.NewScaler("standard")
            result, err := scaler.FitTransform(c.loadedData.X)
            if err != nil {
                XProcessed = c.loadedData.X
            } else {
                XProcessed = result
            }
        }

        splitter := evaluation.NewTrainTestSplitter(1-splitRatio, time.Now().UnixNano(), true)
        XTrain, XTest, yTrain, yTest, err := splitter.StratifiedSplit(XProcessed, c.loadedData.y)
        if err != nil {
            trainErr = fmt.Errorf("failed to split data: %w", err)
            return
        }
        
        trainErr = model.Fit(XTrain, yTrain)
        if trainErr != nil {
            return
        }

        predictions := model.Predict(XTest)
        classes := models.ExtractClasses(c.loadedData.y)
        metrics := evaluation.CalculateMetrics(yTest, predictions, classes)
        
        bundle := persistence.NewModelBundle(model)
        bundle.Scaler = scaler
        bundle.Metadata.Dataset = c.loadedData.SourceFile
        bundle.Metadata.Accuracy = metrics.Accuracy
        bundle.Metadata.Precision = metrics.MacroPrecision
        bundle.Metadata.Recall = metrics.MacroRecall
        bundle.Metadata.F1Score = metrics.MacroF1
        bundle.Metadata.TrainingTime = time.Since(startTime)
        
        timestamp := time.Now().Format("20060102_150405")
        datasetName := strings.TrimSuffix(filepath.Base(c.loadedData.SourceFile), filepath.Ext(c.loadedData.SourceFile))
        modelName := fmt.Sprintf("%s_%s_%s_%s", algorithm, datasetName, prepMethod, timestamp)
        filename := filepath.Join("models", modelName+".model")
        
        os.MkdirAll("models", 0755)
        
        if err := bundle.Save(filename); err != nil {
            trainErr = fmt.Errorf("failed to save model: %w", err)
        } else {
            c.currentModel = model
            c.modelBundle = bundle
            c.currentModelPath = filename

            c.saveTrainingLog(modelName, bundle.Metadata)
        }
    }()
    
    select {
    case <-done:
        stopProgress <- true
        if trainErr != nil {
            fmt.Printf("%s Training error: %v\n", c.red("✗"), trainErr)
            return
        }
        trainingTime := time.Since(startTime)
        fmt.Printf("%s Model trained successfully!\n", c.green("✓"))
        fmt.Println(strings.Repeat("─", 50))
        fmt.Printf("Training time:  %.2fs\n", trainingTime.Seconds())
        fmt.Printf("Test Accuracy:  %.4f\n", c.modelBundle.Metadata.Accuracy)
        fmt.Printf("Test Precision: %.4f\n", c.modelBundle.Metadata.Precision)
        fmt.Printf("Test Recall:    %.4f\n", c.modelBundle.Metadata.Recall)
        fmt.Printf("Test F1 Score:  %.4f\n", c.modelBundle.Metadata.F1Score)
        fmt.Println(strings.Repeat("─", 50))
        fmt.Printf("Model saved: %s\n", c.currentModelPath)

    case <-interrupt:
        stopProgress <- true
        fmt.Printf("\n%s Training interrupted by user\n", c.yellow("⚠"))
        return
    case <-time.After(5 * time.Minute):
        stopProgress <- true
        fmt.Printf("\n%s Training timeout (5 minutes)\n", c.red("✗"))
        return
    }
}

func (c *Commander) evaluate() {
    if c.currentModel == nil {
        fmt.Println(c.red("No model trained. Train a model first"))
        return
    }
    
    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded"))
        return
    }
    
    fmt.Println(c.blue("\nModel Evaluation:"))
    fmt.Println(strings.Repeat("═", 60))
    
    splitter := evaluation.NewTrainTestSplitter(0.2, time.Now().UnixNano(), true)
    _, XTest, _, yTest, err := splitter.StratifiedSplit(c.loadedData.X, c.loadedData.y)
    if err != nil {
        fmt.Printf("%s Error splitting data: %v\n", c.red("✗"), err)
        return
    }

    XTestProcessed := XTest
    if c.modelBundle != nil && c.modelBundle.Scaler != nil {
        XTestProcessed, err = c.modelBundle.Scaler.Transform(XTest)
        if err != nil {
            fmt.Printf("%s Error preprocessing data: %v\n", c.red("✗"), err)
            return
        }
    }

    predictions := c.currentModel.Predict(XTestProcessed)
    classes := models.ExtractClasses(c.loadedData.y)
    metrics := evaluation.CalculateMetrics(yTest, predictions, classes)
    
    fmt.Println(c.cyan("Confusion Matrix:"))
    fmt.Println("(Rows = Actual, Columns = Predicted)")
    fmt.Print("                  ")
    for _, class := range classes {
        className := fmt.Sprintf("C%d", class)
        if c.loadedData.Classes != nil {
            if class < len(c.loadedData.Classes) {
                name := c.loadedData.Classes[class]
                if len(name) > 8 {
                    className = name[:8]
                } else {
                    className = name
                }
            }
        }
        fmt.Printf("%-10s", className)
    }
    fmt.Println()
    
    for _, actualClass := range classes {
        actualName := fmt.Sprintf("Class %d", actualClass)
        if c.loadedData.Classes != nil {
            if actualClass < len(c.loadedData.Classes) {
                actualName = c.loadedData.Classes[actualClass]
            }
        }
        fmt.Printf("%-18s", actualName)
        
        for _, predClass := range classes {
            count := metrics.ConfusionMatrix[actualClass][predClass]
            if actualClass == predClass {
                fmt.Printf("%-10s", c.green(fmt.Sprintf("%d", count)))
            } else if count > 0 {
                fmt.Printf("%-10s", c.red(fmt.Sprintf("%d", count)))
            } else {
                fmt.Printf("%-10d", count)
            }
        }
        fmt.Println()
    }
    
    fmt.Println(strings.Repeat("─", 60))
    
    fmt.Println(c.cyan("\nAccuracy Metrics (Different Calculations):"))
    fmt.Println(strings.Repeat("─", 60))
    
    fmt.Printf("Simple Accuracy:    %.4f  (%d/%d correct)\n", 
        metrics.Accuracy, 
        int(metrics.Accuracy*float64(len(yTest))), 
        len(yTest))
    
    fmt.Printf("Balanced Accuracy:  %.4f  (avg of class recalls)\n", 
        metrics.BalancedAccuracy)
    
    fmt.Printf("Weighted Accuracy:  %.4f  (weighted by class freq)\n", 
        metrics.WeightedAccuracy)
    
    fmt.Println(c.cyan("\nPrecision, Recall, F1 (Different Averaging):"))
    fmt.Println(strings.Repeat("─", 60))
    fmt.Printf("%-15s %-12s %-12s %-12s\n", "Method", "Precision", "Recall", "F1-Score")
    fmt.Println(strings.Repeat("─", 60))
    
    fmt.Printf("%-15s %-12.4f %-12.4f %-12.4f\n",
        "Macro",
        metrics.MacroPrecision,
        metrics.MacroRecall,
        metrics.MacroF1)
    
    fmt.Printf("%-15s %-12.4f %-12.4f %-12.4f\n",
        "Weighted",
        metrics.WeightedPrecision,
        metrics.WeightedRecall,
        metrics.WeightedF1)
    
    fmt.Printf("%-15s %-12.4f %-12.4f %-12.4f\n",
        "Micro",
        metrics.MicroPrecision,
        metrics.MicroRecall,
        metrics.MicroF1)
    
    fmt.Println(c.cyan("\nPer-Class Metrics:"))
    fmt.Println(strings.Repeat("─", 70))
    fmt.Printf("%-15s %-10s %-10s %-10s %-10s %-8s\n", 
        "Class", "Precision", "Recall", "Specificity", "F1-Score", "Support")
    fmt.Println(strings.Repeat("─", 70))
    
    for _, class := range classes {
        className := fmt.Sprintf("Class %d", class)
        if c.loadedData.Classes != nil {
            if class < len(c.loadedData.Classes) {
                name := c.loadedData.Classes[class]
                className = name
            }
        }
        
        classMetrics := metrics.PerClassMetrics[class]
        fmt.Printf("%-15s %-10.4f %-10.4f %-10.4f %-10.4f %-8d\n",
            className,
            classMetrics.Precision,
            classMetrics.Recall,
            classMetrics.Specificity,
            classMetrics.F1Score,
            metrics.ClassSupport[class])
    }
    
    fmt.Println(strings.Repeat("═", 70))
    
    fmt.Println(c.cyan("\nMetric Definitions:"))
    fmt.Println(strings.Repeat("─", 60))
    
    fmt.Println(c.yellow("Accuracy Types:"))
    fmt.Println("  • " + c.green("Simple:") + " (TP + TN) / Total - Standard accuracy")
    fmt.Println("  • " + c.green("Balanced:") + " Average of recall for each class")
    fmt.Println("  • " + c.green("Weighted:") + " Accuracy weighted by class frequency")
    
    fmt.Println(c.yellow("\nClass Metrics:"))
    fmt.Println("  • " + c.green("Precision:") + " TP / (TP + FP) - When predicted positive, how often correct?")
    fmt.Println("  • " + c.green("Recall:") + " TP / (TP + FN) - Of all positive, how many found?")
    fmt.Println("  • " + c.green("Specificity:") + " TN / (TN + FP) - Of all negative, how many correctly identified?")
    fmt.Println("  • " + c.green("F1-Score:") + " 2*(P*R)/(P+R) - Harmonic mean of precision & recall")
    
    fmt.Println(c.yellow("\nAveraging Methods:"))
    fmt.Println("  • " + c.green("Macro:") + " Simple average - treats all classes equally")
    fmt.Println("  • " + c.green("Weighted:") + " Weighted by class frequency - for imbalanced data")
    fmt.Println("  • " + c.green("Micro:") + " Global calculation - aggregates all TP, FP, FN")
    
    accuracyDiff := math.Abs(metrics.Accuracy - metrics.BalancedAccuracy)
    if accuracyDiff > 0.05 {
        fmt.Printf("\n%s Accuracy difference detected (%.3f)\n", 
            c.yellow("Note:"), accuracyDiff)
        if metrics.BalancedAccuracy < metrics.Accuracy {
            fmt.Println("  Simple accuracy higher than balanced - possible class imbalance")
            fmt.Println("  Model may be biased toward majority class")
        }
    }
}

func (c *Commander) crossValidate() {
    if c.currentModel == nil {
        fmt.Println(c.red("No model trained. Train a model first"))
        return
    }
    
    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded"))
        return
    }
    
    fmt.Println("Running 5-fold cross-validation...")
    
    cv := evaluation.NewCrossValidator(5, true)
    scores, mean, std, err := cv.ParallelCrossValidate(
        c.loadedData.X,
        c.loadedData.y,
        c.currentModel,
    )
    if err != nil {
        scores, mean, std, _ = cv.CrossValidateSerial(
            c.loadedData.X,
            c.loadedData.y,
            c.currentModel,
        )
    }
    
    fmt.Printf("%s Cross-validation complete!\n", c.green("✓"))
    fmt.Printf("Scores: %v\n", scores)
    fmt.Printf("Mean: %.4f (±%.4f)\n", mean, std)
}


func (c *Commander) loadModel(filename string) {
    if !strings.Contains(filename, "/") {
        filename = filepath.Join("models", filename)
    }
    
    if !strings.Contains(filename, ".") {
        filename = filename + ".model"
    }
    
    bundle, err := persistence.LoadModelBundle(filename)
    if err != nil {
        fmt.Printf("%s Error loading model: %v\n", c.red("✗"), err)
        fmt.Println("Ensure the file exists in models/ directory")
        return
    }
    
    c.modelBundle = bundle
    c.currentModel = bundle.Model
    c.currentModelPath = filename
    
    fmt.Printf("%s Model loaded successfully!\n", c.green("✓"))
    fmt.Printf("Model: %s\n", bundle.Metadata.ModelName)
    fmt.Printf("Dataset: %s\n", bundle.Metadata.Dataset)
    fmt.Printf("Accuracy: %.4f | F1: %.4f\n", 
        bundle.Metadata.Accuracy, 
        bundle.Metadata.F1Score)
    fmt.Printf("Created: %s\n", bundle.CreatedAt.Format("2006-01-02 15:04:05"))
    fmt.Println("Use 'predict' or 'evaluate' to interact with the model")
}

func (c *Commander) listModels() {
    modelFiles, err := filepath.Glob("models/*.model")
    if err != nil || len(modelFiles) == 0 {
        fmt.Println("No saved models found in models/ directory")
        fmt.Println("Train a model using 'train <algorithm>' command")
        return
    }
    
    fmt.Println(c.blue("\nSaved Models:"))
    fmt.Println(strings.Repeat("─", 70))
    fmt.Printf("%-30s %-10s %-15s %-10s\n", "Filename", "Size", "Modified", "Status")
    fmt.Println(strings.Repeat("─", 70))
    
    for _, file := range modelFiles {
        info, err := os.Stat(file)
        if err != nil {
            continue
        }
        
        status := ""
        if c.currentModelPath == file {
            status = c.cyan("[ACTIVE]")
        }
        
        fmt.Printf("%-30s %-10s %-15s %-10s\n", 
            filepath.Base(file),
            fmt.Sprintf("%.1f KB", float64(info.Size())/1024),
            info.ModTime().Format("01-02 15:04"),
            status)
    }
    
    fmt.Println()
    fmt.Println("Use 'loadmodel <filename>' to load a model")
    if c.currentModel != nil {
        fmt.Printf("Current active model: %s\n", c.currentModel.GetName())
    }
}

func (c *Commander) showDataInfo() {
    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded"))
        return
    }
    
    fmt.Println(c.blue("\nDataset Information:"))
    fmt.Println(strings.Repeat("─", 40))
    fmt.Printf("Source: %s\n", c.loadedData.SourceFile)
    fmt.Printf("Samples: %d\n", len(c.loadedData.X))
    fmt.Printf("Features: %d\n", len(c.loadedData.Features))
    fmt.Printf("Feature names: %v\n", c.loadedData.Features)
    fmt.Printf("Classes: %d\n", len(c.loadedData.Classes))
    fmt.Printf("Class mapping: %v\n", c.loadedData.Classes)
}

func (c *Commander) showTrainHelp() {
    fmt.Println(c.blue("\nTrain Command Usage:"))
    fmt.Println("  train knn [k]           - K-Nearest Neighbors")
    fmt.Println("  train tree [depth]      - Decision Tree")
    fmt.Println("  train forest [n_trees]  - Random Forest")
    fmt.Println("  train bayes             - Naive Bayes")
}

func (c *Commander) runExperiment() {
    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded. Use 'load <file>' first"))
        return
    }

    fmt.Println(c.cyan("\nExperiment Configuration:"))
    fmt.Println("1. Full comparison (all algorithms, all preprocessing)")
    fmt.Println("2. Single algorithm (choose one)")
    fmt.Println("3. Custom selection (choose specific algorithms)")
    fmt.Println("4. Current model only (test current trained model)")

    scanner := bufio.NewScanner(os.Stdin)
    fmt.Print("Select experiment type (1-4): ")
    scanner.Scan()
    choice := strings.TrimSpace(scanner.Text())

    var selectedAlgorithms []string
    var selectedPreprocessing []string

    switch choice {
    case "2":
        fmt.Println("\nSelect algorithm:")
        fmt.Println("1. KNN")
        fmt.Println("2. Decision Tree")
        fmt.Println("3. Random Forest")
        fmt.Println("4. Naive Bayes")
        fmt.Print("Choice: ")
        scanner.Scan()
        algoChoice := strings.TrimSpace(scanner.Text())
        switch algoChoice {
        case "1":
            selectedAlgorithms = []string{"KNN"}
        case "2":
            selectedAlgorithms = []string{"DecisionTree"}
        case "3":
            selectedAlgorithms = []string{"RandomForest"}
        case "4":
            selectedAlgorithms = []string{"NaiveBayes"}
        default:
            fmt.Println(c.red("Invalid choice"))
            return
        }

    case "3":
        fmt.Println("\nSelect algorithms (comma-separated, e.g., 1,3):")
        fmt.Println("1. KNN")
        fmt.Println("2. Decision Tree")
        fmt.Println("3. Random Forest")
        fmt.Println("4. Naive Bayes")
        fmt.Print("Choices: ")
        scanner.Scan()
        choices := strings.Split(scanner.Text(), ",")
        for _, ch := range choices {
            switch strings.TrimSpace(ch) {
            case "1":
                selectedAlgorithms = append(selectedAlgorithms, "KNN")
            case "2":
                selectedAlgorithms = append(selectedAlgorithms, "DecisionTree")
            case "3":
                selectedAlgorithms = append(selectedAlgorithms, "RandomForest")
            case "4":
                selectedAlgorithms = append(selectedAlgorithms, "NaiveBayes")
            }
        }
        if len(selectedAlgorithms) == 0 {
            fmt.Println(c.red("No valid algorithms selected"))
            return
        }

    case "4":
        if c.currentModel == nil {
            fmt.Println(c.red("No model currently trained. Train a model first."))
            return
        }
        c.saveCurrentModelResults(5)
        return

    case "1":
        selectedAlgorithms = []string{"KNN", "DecisionTree", "RandomForest", "NaiveBayes"}

    default:
        fmt.Println(c.red("Invalid choice"))
        return
    }

    if choice != "4" {
        fmt.Println("\nSelect preprocessing methods:")
        fmt.Println("1. All (raw, normalized, standardized)")
        fmt.Println("2. Raw only")
        fmt.Println("3. Normalized only")
        fmt.Println("4. Standardized only")
        fmt.Println("5. Custom selection")
        fmt.Print("Choice: ")
        scanner.Scan()
        prepChoice := strings.TrimSpace(scanner.Text())

        switch prepChoice {
        case "1":
            selectedPreprocessing = []string{"raw", "normalized", "standardized"}
        case "2":
            selectedPreprocessing = []string{"raw"}
        case "3":
            selectedPreprocessing = []string{"normalized"}
        case "4":
            selectedPreprocessing = []string{"standardized"}
        case "5":
            fmt.Print("Enter methods (comma-separated: raw,normalized,standardized): ")
            scanner.Scan()
            selectedPreprocessing = strings.Split(scanner.Text(), ",")
            for i := range selectedPreprocessing {
                selectedPreprocessing[i] = strings.TrimSpace(selectedPreprocessing[i])
            }
        default:
            selectedPreprocessing = []string{"raw", "normalized", "standardized"}
        }
    }

    fmt.Print("\nNumber of CV folds (default 5): ")
    scanner.Scan()
    foldsStr := strings.TrimSpace(scanner.Text())
    folds := 5
    if foldsStr != "" {
        if f, err := strconv.Atoi(foldsStr); err == nil && f > 1 {
            folds = f
        }
    }

    fmt.Printf("\nStarting experiment with:")
    fmt.Printf("\n  Algorithms: %v", selectedAlgorithms)
    fmt.Printf("\n  Preprocessing: %v", selectedPreprocessing)
    fmt.Printf("\n  CV Folds: %d\n\n", folds)

    c.runCustomExperiment(selectedAlgorithms, selectedPreprocessing, folds)
    
    timestamp := time.Now().Format("20060102_150405")
    datasetName := filepath.Base(c.loadedData.SourceFile)
    datasetName = strings.TrimSuffix(datasetName, filepath.Ext(datasetName))

    expDir := fmt.Sprintf("results/exp_%s_%s", datasetName, timestamp)
    os.MkdirAll(expDir, 0755)

    fmt.Printf("Creating experiment directory: %s\n", expDir)
    
}

func (c *Commander) runCustomExperiment(selectedAlgorithms []string, selectedPreprocessing []string, folds int) {
    timestamp := time.Now().Format("20060102_150405")
    datasetName := filepath.Base(c.loadedData.SourceFile)
    datasetName = strings.TrimSuffix(datasetName, filepath.Ext(datasetName))

    expDir := fmt.Sprintf("results/exp_%s_%s", datasetName, timestamp)
    os.MkdirAll(expDir, 0755)

    fmt.Printf("Creating experiment directory: %s\n", expDir)

    allConfigs := map[string][]models.ModelConfig{
        "KNN": {
            {Algorithm: "knn", K: 3, Distance: "euclidean"},
            {Algorithm: "knn", K: 5, Distance: "euclidean"},
            {Algorithm: "knn", K: 7, Distance: "manhattan"},
        },
        "DecisionTree": {
            {Algorithm: "tree", MaxDepth: 5, MinSplit: 2},
            {Algorithm: "tree", MaxDepth: 10, MinSplit: 5},
        },
        "RandomForest": {
            {Algorithm: "forest", NTrees: 10, MaxDepth: 5, MinSplit: 2},
            {Algorithm: "forest", NTrees: 50, MaxDepth: 10, MinSplit: 2},
        },
        "NaiveBayes": {
            {Algorithm: "bayes", VarSmoothing: 1e-9},
        },
    }

    var experimentConfigs []struct {
        name    string
        configs []models.ModelConfig
    }

    for _, algoName := range selectedAlgorithms {
        if configs, ok := allConfigs[algoName]; ok {
            experimentConfigs = append(experimentConfigs, struct {
                name    string
                configs []models.ModelConfig
            }{algoName, configs})
        }
    }

    var algorithms []struct {
        name   string
        models []models.Model
    }

    for _, expConfig := range experimentConfigs {
        var modelList []models.Model
        for _, config := range expConfig.configs {
            model, err := models.CreateModel(config)
            if err != nil {
                fmt.Printf("%s Failed to create %s model: %v\n", c.red("✗"), config.Algorithm, err)
                continue
            }
            modelList = append(modelList, model)
        }
        if len(modelList) > 0 {
            algorithms = append(algorithms, struct {
                name   string
                models []models.Model
            }{expConfig.name, modelList})
        }
    }
    
    resultsFile, _ := os.Create(filepath.Join(expDir, "experiment_results.csv"))
    defer resultsFile.Close()
    
    writer := csv.NewWriter(resultsFile)
    writer.Write([]string{
        "Algorithm", "Parameters", "Preprocessing", "TrainTestSplit",
        "Accuracy", "Precision", "Recall", "F1Score", "CVMean", "CVStd",
    })
    
    evalDataFile, _ := os.Create(filepath.Join(expDir, "evaluation_data.csv"))
    defer evalDataFile.Close()
    evalWriter := csv.NewWriter(evalDataFile)
    evalWriter.Write([]string{
        "Algorithm", "Preprocessing", "Split", "SampleIndex", "OriginalIndex",
        "Features", "ActualClass", "PredictedClass", "Correct", "Confidence",
    })
    
    cvFoldsFile, _ := os.Create(filepath.Join(expDir, "cv_folds.csv"))
    defer cvFoldsFile.Close()
    cvWriter := csv.NewWriter(cvFoldsFile)
    cvWriter.Write([]string{
        "Algorithm", "Preprocessing", "Fold", "Type", "SampleIndex", "OriginalIndex", "Class",
    })

    cvPerfFile, _ := os.Create(filepath.Join(expDir, "cv_performance.csv"))
    defer cvPerfFile.Close()
    cvPerfWriter := csv.NewWriter(cvPerfFile)
    cvPerfWriter.Write([]string{
        "Algorithm", "Preprocessing", "Fold", "Accuracy", "Precision", "Recall", "F1",
        "TestSize", "TrainSize", "Correct", "Total",
    })
    
    splitsFile, _ := os.Create(filepath.Join(expDir, "data_splits.csv"))
    defer splitsFile.Close()
    splitWriter := csv.NewWriter(splitsFile)
    splitWriter.Write([]string{
        "Algorithm", "Preprocessing", "Split", "Type", "SampleIndex", "OriginalIndex", 
        "Features", "Class",
    })
    
    cv := evaluation.NewCrossValidator(folds, true)

    preprocessingMethods := selectedPreprocessing
    
    for _, prepMethod := range preprocessingMethods {
        XProcessed := c.loadedData.X
        var scaler *preprocessing.Scaler
        
        switch prepMethod {
        case "normalized":
            scaler = preprocessing.NewScaler("minmax")
            result, err := scaler.FitTransform(c.loadedData.X)
            if err != nil {
                fmt.Printf("%s Preprocessing failed, using raw data\n", c.yellow("⚠"))
                XProcessed = c.loadedData.X
            } else {
                XProcessed = result
            }
        case "standardized":
            scaler = preprocessing.NewScaler("standard")
            result, err := scaler.FitTransform(c.loadedData.X)
            if err != nil {
                fmt.Printf("%s Preprocessing failed, using raw data\n", c.yellow("⚠"))
                XProcessed = c.loadedData.X
            } else {
                XProcessed = result
            }
        }
        
        for _, algo := range algorithms {
            fmt.Printf("\nTesting %s models with %s preprocessing...\n", algo.name, prepMethod)
            
            for _, model := range algo.models {
                splitIdx := int(float64(len(XProcessed)) * 0.8)
                XTrain := XProcessed[:splitIdx]
                XTest := XProcessed[splitIdx:]
                yTrain := c.loadedData.y[:splitIdx]
                yTest := c.loadedData.y[splitIdx:]
                
                for i := 0; i < splitIdx; i++ {
                    features := make([]string, len(XTrain[i]))
                    for j, f := range XTrain[i] {
                        features[j] = f.String()
                    }
                    
                    className := fmt.Sprintf("%d", yTrain[i])
                    if c.loadedData.Classes != nil {
                        if yTrain[i] < len(c.loadedData.Classes) {
                            className = c.loadedData.Classes[yTrain[i]]
                        }
                    }
                    
                    splitWriter.Write([]string{
                        model.GetName(),
                        prepMethod,
                        "80-20",
                        "Train",
                        fmt.Sprintf("%d", i),
                        fmt.Sprintf("%d", i),
                        strings.Join(features, ";"),
                        className,
                    })
                }
                
                for i := 0; i < len(XTest); i++ {
                    features := make([]string, len(XTest[i]))
                    for j, f := range XTest[i] {
                        features[j] = f.String()
                    }
                    
                    className := fmt.Sprintf("%d", yTest[i])
                    if c.loadedData.Classes != nil {
                        if yTest[i] < len(c.loadedData.Classes) {
                            className = c.loadedData.Classes[yTest[i]]
                        }
                    }
                    
                    splitWriter.Write([]string{
                        model.GetName(),
                        prepMethod,
                        "80-20",
                        "Test",
                        fmt.Sprintf("%d", i),
                        fmt.Sprintf("%d", splitIdx+i),
                        strings.Join(features, ";"),
                        className,
                    })
                }
                
                model.Fit(XTrain, yTrain)
                predictions := model.Predict(XTest)
                probabilities := model.PredictProba(XTest)
                
                for i := range XTest {
                    features := make([]string, len(XTest[i]))
                    for j, f := range XTest[i] {
                        features[j] = f.String()
                    }
                    
                    actualClass := fmt.Sprintf("%d", yTest[i])
                    predictedClass := fmt.Sprintf("%d", predictions[i])
                    if c.loadedData.Classes != nil {
                        if yTest[i] < len(c.loadedData.Classes) {
                            actualClass = c.loadedData.Classes[yTest[i]]
                        }
                        if predictions[i] < len(c.loadedData.Classes) {
                            predictedClass = c.loadedData.Classes[predictions[i]]
                        }
                    }
                    
                    correct := "0"
                    if yTest[i] == predictions[i] {
                        correct = "1"
                    }
                    
                    confidence := "0"
                    if len(probabilities) > i && predictions[i] < len(probabilities[i]) {
                        f, _ := probabilities[i][predictions[i]].Float64()
                        confidence = fmt.Sprintf("%.4f", f)
                    }
                    
                    evalWriter.Write([]string{
                        model.GetName(),
                        prepMethod,
                        "80-20",
                        fmt.Sprintf("%d", i),
                        fmt.Sprintf("%d", splitIdx+i),
                        strings.Join(features, ";"),
                        actualClass,
                        predictedClass,
                        correct,
                        confidence,
                    })
                }
                
                classes := models.ExtractClasses(c.loadedData.y)
                metrics := evaluation.CalculateMetrics(yTest, predictions, classes)
                
                folds, err := cv.KFoldSplit(len(XProcessed), c.loadedData.y)
                if err != nil {
                    fmt.Printf("%s Cross-validation failed: %v\n", c.red("✗"), err)
                    continue
                }
                cvScores := make([]float64, cv.NFolds)
                
                for foldIdx, testIndices := range folds {
                    testIndicesMap := make(map[int]bool)
                    for _, idx := range testIndices {
                        testIndicesMap[idx] = true
                    }
                    
                    var XTrainCV, XTestCV [][]decimal.Decimal
                    var yTrainCV, yTestCV []int
                    
                    for j := 0; j < len(XProcessed); j++ {
                        className := fmt.Sprintf("%d", c.loadedData.y[j])
                        if c.loadedData.Classes != nil {
                            if c.loadedData.y[j] < len(c.loadedData.Classes) {
                                className = c.loadedData.Classes[c.loadedData.y[j]]
                            }
                        }
                        
                        if testIndicesMap[j] {
                            XTestCV = append(XTestCV, XProcessed[j])
                            yTestCV = append(yTestCV, c.loadedData.y[j])
                            
                            cvWriter.Write([]string{
                                model.GetName(),
                                prepMethod,
                                fmt.Sprintf("%d", foldIdx+1),
                                "Test",
                                fmt.Sprintf("%d", len(XTestCV)-1),
                                fmt.Sprintf("%d", j),
                                className,
                            })
                        } else {
                            XTrainCV = append(XTrainCV, XProcessed[j])
                            yTrainCV = append(yTrainCV, c.loadedData.y[j])
                            
                            cvWriter.Write([]string{
                                model.GetName(),
                                prepMethod,
                                fmt.Sprintf("%d", foldIdx+1),
                                "Train",
                                fmt.Sprintf("%d", len(XTrainCV)-1),
                                fmt.Sprintf("%d", j),
                                className,
                            })
                        }
                    }
                    
                    model.Reset()
                    model.Fit(XTrainCV, yTrainCV)
                    foldPredictions := model.Predict(XTestCV)
                    
                    correct := 0
                    for k := range foldPredictions {
                        if foldPredictions[k] == yTestCV[k] {
                            correct++
                        }
                    }
                    foldAccuracy := float64(correct) / float64(len(yTestCV))
                    cvScores[foldIdx] = foldAccuracy

                    foldClasses := models.ExtractClasses(yTestCV)
                    foldMetrics := evaluation.CalculateMetrics(yTestCV, foldPredictions, foldClasses)

                    cvPerfWriter.Write([]string{
                        model.GetName(),
                        prepMethod,
                        fmt.Sprintf("%d", foldIdx+1),
                        fmt.Sprintf("%.4f", foldAccuracy),
                        fmt.Sprintf("%.4f", foldMetrics.MacroPrecision),
                        fmt.Sprintf("%.4f", foldMetrics.MacroRecall),
                        fmt.Sprintf("%.4f", foldMetrics.MacroF1),
                        fmt.Sprintf("%d", len(yTestCV)),
                        fmt.Sprintf("%d", len(yTrainCV)),
                        fmt.Sprintf("%d", correct),
                        fmt.Sprintf("%d", len(yTestCV)),
                    })
                }
                
                cvMean := 0.0
                for _, score := range cvScores {
                    cvMean += score
                }
                cvMean /= float64(len(cvScores))
                
                cvVariance := 0.0
                for _, score := range cvScores {
                    diff := score - cvMean
                    cvVariance += diff * diff
                }
                cvVariance /= float64(len(cvScores))
                cvStd := math.Sqrt(cvVariance)
                
                writer.Write([]string{
                    algo.name,
                    fmt.Sprintf("%v", model.GetParams()),
                    prepMethod,
                    "80-20",
                    fmt.Sprintf("%.4f", metrics.Accuracy),
                    fmt.Sprintf("%.4f", metrics.MacroPrecision),
                    fmt.Sprintf("%.4f", metrics.MacroRecall),
                    fmt.Sprintf("%.4f", metrics.MacroF1),
                    fmt.Sprintf("%.4f", cvMean),
                    fmt.Sprintf("%.4f", cvStd),
                })
                
                fmt.Printf("  %s: Acc=%.4f, F1=%.4f, CV=%.4f(±%.4f)\n", 
                    model.GetName(), 
                    metrics.Accuracy,
                    metrics.MacroF1,
                    cvMean, cvStd)
            }
        }
    }
    
    writer.Flush()
    evalWriter.Flush()
    cvWriter.Flush()
    cvPerfWriter.Flush()
    splitWriter.Flush()
    
    summaryFile, _ := os.Create(filepath.Join(expDir, "experiment_summary.txt"))
    defer summaryFile.Close()
    
    fmt.Fprintf(summaryFile, "Experiment Summary\n")
    fmt.Fprintf(summaryFile, "==================\n\n")
    fmt.Fprintf(summaryFile, "Timestamp: %s\n", timestamp)
    fmt.Fprintf(summaryFile, "Dataset: %s\n", c.loadedData.SourceFile)
    fmt.Fprintf(summaryFile, "Total Samples: %d\n", len(c.loadedData.X))
    fmt.Fprintf(summaryFile, "Features: %d\n", len(c.loadedData.Features))
    fmt.Fprintf(summaryFile, "Classes: %d\n\n", len(c.loadedData.Classes))
    
    fmt.Fprintf(summaryFile, "Files Generated:\n")
    fmt.Fprintf(summaryFile, "1. experiment_results.csv - Main results with metrics\n")
    fmt.Fprintf(summaryFile, "2. evaluation_data.csv - Detailed predictions for each sample\n")
    fmt.Fprintf(summaryFile, "3. cv_folds.csv - Cross-validation fold assignments\n")
    fmt.Fprintf(summaryFile, "4. cv_performance.csv - Performance metrics per CV fold\n")
    fmt.Fprintf(summaryFile, "5. data_splits.csv - Train/test split details\n")
    
    fmt.Printf("\n%s Experiment complete! Results saved to:\n", c.green("✓"))
    fmt.Printf("  • Directory: %s\n", expDir)
    fmt.Printf("  • Main results: %s\n", filepath.Join(expDir, "experiment_results.csv"))
    fmt.Printf("  • Evaluation data: %s\n", filepath.Join(expDir, "evaluation_data.csv"))
    fmt.Printf("  • CV performance: %s\n", filepath.Join(expDir, "cv_performance.csv"))
    fmt.Printf("  • CV folds: %s\n", filepath.Join(expDir, "cv_folds.csv"))
    fmt.Printf("  • Data splits: %s\n", filepath.Join(expDir, "data_splits.csv"))
    fmt.Printf("  • Summary: %s\n", filepath.Join(expDir, "experiment_summary.txt"))
}

func (c *Commander) predict(args []string) {
    if c.currentModel == nil {
        fmt.Println(c.red("No model loaded. Train or load a model first"))
        return
    }
    
    if c.loadedData == nil || c.loadedData.Features == nil {
        fmt.Println(c.red("No dataset information available"))
        fmt.Println("Please load the same dataset used for training")
        return
    }
    
    expectedFeatures := len(c.loadedData.Features)
    
    if len(args) > 0 {
        if len(args) < expectedFeatures {
            fmt.Printf("%s Error: Too few values provided\n", c.red("✗"))
            fmt.Printf("Expected %d values for features: %v\n", 
                expectedFeatures, c.loadedData.Features)
            fmt.Printf("You provided only %d values: %v\n", len(args), args)
            fmt.Println("\nUsage examples:")
            fmt.Printf("  predict %s\n", strings.Join(c.generateSampleValues(), " "))
            fmt.Println("  predict  (for interactive mode)")
            return
        }
        
        if len(args) > expectedFeatures {
            fmt.Printf("%s Warning: Too many values provided\n", c.yellow("⚠"))
            fmt.Printf("Expected %d values, got %d. Using first %d values.\n", 
                expectedFeatures, len(args), expectedFeatures)
            args = args[:expectedFeatures]
        }
        
        sample := make([]decimal.Decimal, expectedFeatures)
        for i := 0; i < expectedFeatures; i++ {
            val, err := decimal.NewFromString(args[i])
            if err != nil {
                fmt.Printf("%s Invalid value for %s: %s\n", 
                    c.red("✗"), c.loadedData.Features[i], args[i])
                fmt.Println("Values must be numeric")
                return
            }
            sample[i] = val
        }
        
        c.makePrediction(sample)
        return
    }
    
    fmt.Println(c.cyan("Enter values for prediction:"))
    fmt.Printf("Features (%d required): %v\n", expectedFeatures, c.loadedData.Features)
    
    if len(c.loadedData.X) > 0 {
        fmt.Println("\nValid ranges from dataset:")
        for i, feature := range c.loadedData.Features {
            min := c.loadedData.X[0][i]
            max := c.loadedData.X[0][i]
            sum := decimal.Zero
            for _, row := range c.loadedData.X {
                if row[i].LessThan(min) {
                    min = row[i]
                }
                if row[i].GreaterThan(max) {
                    max = row[i]
                }
                sum = sum.Add(row[i])
            }
            avg := sum.Div(decimal.NewFromInt(int64(len(c.loadedData.X))))
            minF, _ := min.Float64()
            maxF, _ := max.Float64()
            avgF, _ := avg.Float64()
            fmt.Printf("  %s: %.2f to %.2f (avg: %.2f)\n", 
                feature, minF, maxF, avgF)
        }
        
        fmt.Printf("\nExample: %s\n", strings.Join(c.generateSampleValues(), ", "))
    }
    
    fmt.Println("\n" + c.yellow("Enter values:"))
    fmt.Println("  Option 1: All values comma-separated (e.g., 5.1, 3.5, 1.4, 0.2)")
    fmt.Println("  Option 2: Press Enter to input each value separately")
    
    scanner := bufio.NewScanner(os.Stdin)
    sample := make([]decimal.Decimal, expectedFeatures)
    
    fmt.Print("→ ")
    scanner.Scan()
    input := strings.TrimSpace(scanner.Text())
    
    if strings.Contains(input, ",") {
        values := strings.Split(input, ",")
        
        if len(values) < expectedFeatures {
            fmt.Printf("%s Error: Too few values\n", c.red("✗"))
            fmt.Printf("Expected %d values, got %d\n", expectedFeatures, len(values))
            fmt.Printf("Missing features: %v\n", c.loadedData.Features[len(values):])
            return
        }
        
        if len(values) > expectedFeatures {
            fmt.Printf("%s Warning: Too many values, using first %d\n", 
                c.yellow("⚠"), expectedFeatures)
            values = values[:expectedFeatures]
        }
        
        for i, v := range values[:expectedFeatures] {
            val, err := decimal.NewFromString(strings.TrimSpace(v))
            if err != nil {
                fmt.Printf("%s Invalid value for %s: %s\n", 
                    c.red("✗"), c.loadedData.Features[i], v)
                fmt.Println("Please enter numeric values only")
                return
            }
            sample[i] = val
        }
    } else if input != "" {
        val, err := decimal.NewFromString(input)
        if err != nil {
            fmt.Printf("%s Invalid value for %s: %s\n", 
                c.red("✗"), c.loadedData.Features[0], input)
            return
        }
        sample[0] = val
        fmt.Printf("  %s = %s ✓\n", c.loadedData.Features[0], val)
        
        for i := 1; i < expectedFeatures; i++ {
            fmt.Printf("%s (%d/%d): ", c.loadedData.Features[i], i+1, expectedFeatures)
            scanner.Scan()
            input := strings.TrimSpace(scanner.Text())
            
            if input == "" {
                fmt.Printf("%s Error: Value required for %s\n", 
                    c.red("✗"), c.loadedData.Features[i])
                fmt.Printf("Still need %d more values\n", expectedFeatures-i)
                return
            }
            
            val, err := decimal.NewFromString(input)
            if err != nil {
                fmt.Printf("%s Invalid value: %s\n", c.red("✗"), input)
                fmt.Println("Please enter a numeric value")
                i--
                continue
            }
            sample[i] = val
        }
    } else {
        for i := 0; i < expectedFeatures; i++ {
            fmt.Printf("%s (%d/%d): ", c.loadedData.Features[i], i+1, expectedFeatures)
            scanner.Scan()
            input := strings.TrimSpace(scanner.Text())
            
            if input == "" {
                fmt.Printf("%s Error: Value required for %s\n", 
                    c.red("✗"), c.loadedData.Features[i])
                i--
                continue
            }
            
            val, err := decimal.NewFromString(input)
            if err != nil {
                fmt.Printf("%s Invalid value: %s\n", c.red("✗"), input)
                fmt.Println("Please enter a numeric value")
                i--
                continue
            }
            sample[i] = val
        }
    }
    
    c.makePrediction(sample)
}

func (c *Commander) generateSampleValues() []string {
    if c.loadedData == nil || len(c.loadedData.X) == 0 {
        return []string{}
    }
    
    values := make([]string, len(c.loadedData.Features))
    for i := range c.loadedData.Features {
        sum := decimal.Zero
        for _, row := range c.loadedData.X {
            sum = sum.Add(row[i])
        }
        avg := sum.Div(decimal.NewFromInt(int64(len(c.loadedData.X))))
        avgF, _ := avg.Float64()
        values[i] = fmt.Sprintf("%.2f", avgF)
    }
    
    return values
}

func (c *Commander) makePrediction(sample []decimal.Decimal) {
    processedSample := [][]decimal.Decimal{sample}
    
    if c.modelBundle != nil && c.modelBundle.Scaler != nil {
        result, err := c.modelBundle.Scaler.Transform(processedSample)
        if err != nil {
            fmt.Printf("Warning: Preprocessing failed, using raw data: %v\n", err)
        } else {
            processedSample = result
        }
    }
    
    prediction := c.currentModel.Predict(processedSample)
    proba := c.currentModel.PredictProba(processedSample)
    
    fmt.Println("\n" + strings.Repeat("═", 50))
    fmt.Println(c.green("Prediction Results:"))
    fmt.Println(strings.Repeat("─", 50))
    
    fmt.Println("Input values:")
    for i, feature := range c.loadedData.Features {
        value, _ := sample[i].Float64()
        fmt.Printf("  %s: %.4f\n", feature, value)
    }
    
    fmt.Println(strings.Repeat("─", 50))
    
    predictedClass := prediction[0]
    className := fmt.Sprintf("Class %d", predictedClass)
    if c.loadedData.Classes != nil {
        if predictedClass < len(c.loadedData.Classes) {
            className = c.loadedData.Classes[predictedClass]
        }
    }
    
    fmt.Printf("Predicted Class: %s\n", c.cyan(className))
    
    fmt.Println("\nConfidence Scores:")
    maxProba := decimal.Zero

    modelClasses := c.currentModel.GetClasses()

    for i, p := range proba[0] {
        actualClass := modelClasses[i]
        className := fmt.Sprintf("Class %d", actualClass)
        if c.loadedData.Classes != nil {
            if actualClass < len(c.loadedData.Classes) {
                className = c.loadedData.Classes[actualClass]
            }
        }

        barLength := int(p.Mul(decimal.NewFromInt(30)).IntPart())
        bar := strings.Repeat("█", barLength) + strings.Repeat("░", 30-barLength)

        color := c.yellow
        if actualClass == predictedClass {
            color = c.green
            maxProba = p
        }

        f, _ := p.Mul(decimal.NewFromInt(100)).Float64()
        fmt.Printf("  %s: %s %.2f%%\n",
            color(fmt.Sprintf("%-15s", className)),
            bar,
            f)
    }
    
    fmt.Println(strings.Repeat("═", 50))
    
    confidence, _ := maxProba.Float64()
    if confidence > 0.9 {
        fmt.Printf("Confidence Level: %s (%.2f%%)\n",
            c.green("Very High"), confidence*100)
    } else if confidence > 0.7 {
        fmt.Printf("Confidence Level: %s (%.2f%%)\n",
            c.green("High"), confidence*100)
    } else if confidence > 0.5 {
        fmt.Printf("Confidence Level: %s (%.2f%%)\n",
            c.yellow("Moderate"), confidence*100)
    } else {
        fmt.Printf("Confidence Level: %s (%.2f%%)\n",
            c.red("Low"), confidence*100)
    }
}

func (c *Commander) interactiveEvaluate() {
    if c.currentModel == nil {
        fmt.Println(c.red("No model loaded. Train or load a model first"))
        return
    }
    
    fmt.Println(c.cyan("Interactive Evaluation Mode"))
    fmt.Println("Enter 'done' when finished, 'help' for commands")
    
    correctPredictions := 0
    totalPredictions := 0
    
    scanner := bufio.NewScanner(os.Stdin)
    
    for {
        fmt.Println("\n" + strings.Repeat("─", 40))
        fmt.Print("Enter values (or command): ")
        scanner.Scan()
        input := strings.TrimSpace(scanner.Text())
        
        if input == "done" || input == "exit" {
            break
        }
        
        if input == "help" {
            fmt.Println("Commands:")
            fmt.Println("  <values>  - Comma-separated feature values")
            fmt.Println("  random    - Use random sample from dataset")
            fmt.Println("  done      - Exit evaluation mode")
            continue
        }
        
        if input == "random" {
            if c.loadedData == nil || len(c.loadedData.X) == 0 {
                fmt.Println("No dataset loaded")
                continue
            }
            
            idx := rand.Intn(len(c.loadedData.X))
            sample := c.loadedData.X[idx]
            actualClass := c.loadedData.y[idx]
            
            fmt.Println("Random sample selected:")
            for i, feature := range c.loadedData.Features {
                val, _ := sample[i].Float64()
                fmt.Printf("  %s: %.4f\n", feature, val)
            }
            
            processedSample := [][]decimal.Decimal{sample}
            if c.modelBundle != nil && c.modelBundle.Scaler != nil {
                result, err := c.modelBundle.Scaler.Transform(processedSample)
                if err != nil {
                    fmt.Printf("Warning: Preprocessing failed, using raw data: %v\n", err)
                } else {
                    processedSample = result
                }
            }
            
            prediction := c.currentModel.Predict(processedSample)[0]
            
            actualName := fmt.Sprintf("Class %d", actualClass)
            predictedName := fmt.Sprintf("Class %d", prediction)
            if c.loadedData.Classes != nil {
                if actualClass < len(c.loadedData.Classes) {
                    actualName = c.loadedData.Classes[actualClass]
                }
                if prediction < len(c.loadedData.Classes) {
                    predictedName = c.loadedData.Classes[prediction]
                }
            }
            
            fmt.Printf("Actual: %s\n", actualName)
            fmt.Printf("Predicted: %s\n", predictedName)
            
            if prediction == actualClass {
                fmt.Println(c.green("✓ Correct!"))
                correctPredictions++
            } else {
                fmt.Println(c.red("✗ Incorrect"))
            }
            totalPredictions++
            
            continue
        }
        
        values := strings.Split(input, ",")
        if len(values) != len(c.loadedData.Features) {
            fmt.Printf("Expected %d values, got %d\n", 
                len(c.loadedData.Features), len(values))
            continue
        }
        
        sample := make([]decimal.Decimal, len(values))
        valid := true
        for i, v := range values {
            val, err := decimal.NewFromString(strings.TrimSpace(v))
            if err != nil {
                fmt.Printf("Invalid value: %s\n", v)
                valid = false
                break
            }
            sample[i] = val
        }
        
        if !valid {
            continue
        }
        
        c.makePrediction(sample)
        
        fmt.Print("\nWhat was the actual class? ")
        scanner.Scan()
        actualInput := strings.TrimSpace(scanner.Text())
        
        var actualClass int
        if val, err := strconv.Atoi(actualInput); err == nil {
            actualClass = val
        } else {
            found := false
            for classID, className := range c.loadedData.Classes {
                if strings.EqualFold(className, actualInput) {
                    actualClass = classID
                    found = true
                    break
                }
            }
            if !found {
                fmt.Println("Unknown class")
                continue
            }
        }
        
        prediction := c.currentModel.Predict([][]decimal.Decimal{sample})[0]
        if prediction == actualClass {
            fmt.Println(c.green("✓ Model was correct!"))
            correctPredictions++
        } else {
            fmt.Println(c.red("✗ Model was incorrect"))
        }
        totalPredictions++
    }
    
    if totalPredictions > 0 {
        accuracy := float64(correctPredictions) / float64(totalPredictions) * 100
        fmt.Printf("\n%s Interactive Evaluation Summary:\n", c.blue("═"))
        fmt.Printf("Total Predictions: %d\n", totalPredictions)
        fmt.Printf("Correct: %d\n", correctPredictions)
        fmt.Printf("Accuracy: %.2f%%\n", accuracy)
    }
}

func (c *Commander) clearScreen() {
    fmt.Print("\033[H\033[2J")
    c.printWelcome()
}

func (c *Commander) showCurrentModel() {
    if c.currentModel == nil {
        fmt.Println(c.red("No model currently loaded"))
        fmt.Println("Use 'train <algorithm>' to train a new model")
        fmt.Println("Or 'loadmodel <filename>' to load an existing model")
        return
    }
    
    fmt.Println(c.blue("\nCurrent Active Model:"))
    fmt.Println(strings.Repeat("─", 50))
    fmt.Printf("Model Type: %s\n", c.currentModel.GetName())
    fmt.Printf("Parameters: %v\n", c.currentModel.GetParams())
    
    if c.modelBundle != nil {
        fmt.Printf("Dataset: %s\n", c.modelBundle.Metadata.Dataset)
        fmt.Printf("Accuracy: %.4f\n", c.modelBundle.Metadata.Accuracy)
        fmt.Printf("Precision: %.4f\n", c.modelBundle.Metadata.Precision)
        fmt.Printf("Recall: %.4f\n", c.modelBundle.Metadata.Recall)
        fmt.Printf("F1 Score: %.4f\n", c.modelBundle.Metadata.F1Score)
        fmt.Printf("Training Time: %.2fs\n", c.modelBundle.Metadata.TrainingTime.Seconds())
    }
    
    if c.currentModelPath != "" {
        fmt.Printf("File: %s\n", c.currentModelPath)
    }
}

func (c *Commander) batchPredict(filename string) {
    if c.currentModel == nil {
        fmt.Println(c.red("No model loaded. Train or load a model first"))
        return
    }
    
    file, err := os.Open(filename)
    if err != nil {
        fmt.Printf("%s Error opening file: %v\n", c.red("✗"), err)
        return
    }
    defer file.Close()
    
    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        fmt.Printf("%s Error reading CSV: %v\n", c.red("✗"), err)
        return
    }
    
    if len(records) < 2 {
        fmt.Printf("%s No data found in file\n", c.red("✗"))
        return
    }
    
    data := records[1:]
    X := make([][]decimal.Decimal, len(data))
    
    for i, record := range data {
        X[i] = make([]decimal.Decimal, len(record))
        for j, val := range record {
            X[i][j], _ = decimal.NewFromString(val)
        }
    }
    
    fmt.Printf("Making predictions for %d samples...\n", len(X))

    XProcessed := X
    if c.modelBundle != nil && c.modelBundle.Scaler != nil {
        result, err := c.modelBundle.Scaler.Transform(X)
        if err != nil {
            fmt.Printf("Warning: Preprocessing failed, using raw data: %v\n", err)
        } else {
            XProcessed = result
        }
    }

    predictions := c.currentModel.Predict(XProcessed)
    probas := c.currentModel.PredictProba(XProcessed)
    
    outputFile := strings.TrimSuffix(filename, filepath.Ext(filename)) + "_predictions.csv"
    output, err := os.Create(outputFile)
    if err != nil {
        fmt.Printf("%s Error creating output file: %v\n", c.red("✗"), err)
        return
    }
    defer output.Close()
    
    writer := csv.NewWriter(output)
    
    header := []string{"Sample", "Prediction", "Confidence"}
    writer.Write(header)
    
    modelClasses := c.currentModel.GetClasses()

    for i, pred := range predictions {
        confidence := decimal.Zero
        for idx, classID := range modelClasses {
            if classID == pred && idx < len(probas[i]) {
                confidence = probas[i][idx]
                break
            }
        }

        predClass := fmt.Sprintf("%d", pred)
        if c.loadedData != nil && c.loadedData.Classes != nil {
            if pred < len(c.loadedData.Classes) {
                predClass = c.loadedData.Classes[pred]
            }
        }

        writer.Write([]string{
            fmt.Sprintf("%d", i+1),
            predClass,
            fmt.Sprintf("%.4f", func() float64 { f, _ := confidence.Float64(); return f }()),
        })
    }
    
    writer.Flush()
    fmt.Printf("%s Predictions saved to %s\n", c.green("✓"), outputFile)
}

func (c *Commander) compareModels() {
    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded. Load data first with 'load <file>'"))
        return
    }
    
    fmt.Println(c.cyan("Comparing all available models..."))
    
    modelFiles, err := filepath.Glob("models/*.model")
    if err != nil || len(modelFiles) == 0 {
        fmt.Println("No saved models found for comparison")
        return
    }
    
    type ModelComparison struct {
        Name      string
        Accuracy  float64
        F1Score   float64
        Precision float64
        Recall    float64
    }
    
    var comparisons []ModelComparison
    
    for _, modelPath := range modelFiles {
        bundle, err := persistence.LoadModelBundle(modelPath)
        if err != nil {
            continue
        }
        
        comp := ModelComparison{
            Name:      filepath.Base(modelPath),
            Accuracy:  bundle.Metadata.Accuracy,
            F1Score:   bundle.Metadata.F1Score,
            Precision: bundle.Metadata.Precision,
            Recall:    bundle.Metadata.Recall,
        }
        comparisons = append(comparisons, comp)
    }
    
    fmt.Println(c.blue("\nModel Comparison Results:"))
    fmt.Println(strings.Repeat("─", 80))
    fmt.Printf("%-30s %-10s %-10s %-10s %-10s\n", 
        "Model", "Accuracy", "F1", "Precision", "Recall")
    fmt.Println(strings.Repeat("─", 80))
    
    bestAccuracy := 0.0
    bestModel := ""
    
    for _, comp := range comparisons {
        fmt.Printf("%-30s %-10.4f %-10.4f %-10.4f %-10.4f\n",
            comp.Name, comp.Accuracy, comp.F1Score, comp.Precision, comp.Recall)
        
        if comp.Accuracy > bestAccuracy {
            bestAccuracy = comp.Accuracy
            bestModel = comp.Name
        }
    }
    
    fmt.Println(strings.Repeat("─", 80))
    fmt.Printf("\n%s Best model: %s (Accuracy: %.4f)\n",
        c.green("★"), bestModel, bestAccuracy)
}

func (c *Commander) mapToSlice(m map[int]string) []string {
    result := make([]string, len(m))
    for key, value := range m {
        if key < len(result) {
            result[key] = value
        }
    }
    return result
}

func (c *Commander) findBestModel() {
    fmt.Println(c.cyan("Finding best model based on current metrics..."))

    if c.currentModel == nil {
        fmt.Printf("%s No model trained yet. Use 'train' command first.\n", c.red("✗"))
        return
    }

    fmt.Printf("%s Current model: %s\n", c.green("★"), c.currentModel.GetName())
    fmt.Println("To compare multiple models, use 'experiment' command.")
}

func (c *Commander) saveCurrentModelResults(cvFolds int) {
    if c.currentModel == nil {
        fmt.Println(c.red("No model currently trained. Train a model first."))
        return
    }

    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded"))
        return
    }

    fmt.Println(c.cyan("Saving current model evaluation results..."))

    timestamp := time.Now().Format("20060102_150405")
    modelName := c.currentModel.GetName()
    datasetName := filepath.Base(c.loadedData.SourceFile)
    datasetName = strings.TrimSuffix(datasetName, filepath.Ext(datasetName))

    resultsDir := fmt.Sprintf("results/single_%s_%s_%s", modelName, datasetName, timestamp)
    os.MkdirAll(resultsDir, 0755)

    splitter := evaluation.NewTrainTestSplitter(0.2, time.Now().UnixNano(), true)
    XTrain, XTest, yTrain, yTest, err := splitter.StratifiedSplit(c.loadedData.X, c.loadedData.y)
    if err != nil {
        fmt.Printf("%s Error splitting data: %v\n", c.red("✗"), err)
        return
    }

    XTestProcessed := XTest
    XTrainProcessed := XTrain
    prepMethod := "raw"
    if c.modelBundle != nil && c.modelBundle.Scaler != nil {
        XTestProcessed, _ = c.modelBundle.Scaler.Transform(XTest)
        XTrainProcessed, _ = c.modelBundle.Scaler.Transform(XTrain)
        prepMethod = "preprocessed"
    }

    c.currentModel.Fit(XTrainProcessed, yTrain)

    predictions := c.currentModel.Predict(XTestProcessed)
    probas := c.currentModel.PredictProba(XTestProcessed)
    modelClasses := c.currentModel.GetClasses()
    allClasses := models.ExtractClasses(c.loadedData.y)
    metrics := evaluation.CalculateMetrics(yTest, predictions, allClasses)

    fmt.Printf("Running %d-fold cross-validation...\n", cvFolds)
    cv := evaluation.NewCrossValidator(cvFolds, true)

    XProcessed := c.loadedData.X
    if c.modelBundle != nil && c.modelBundle.Scaler != nil {
        XProcessed, _ = c.modelBundle.Scaler.Transform(c.loadedData.X)
    }

    cvPerfFile, _ := os.Create(filepath.Join(resultsDir, "cv_performance.csv"))
    defer cvPerfFile.Close()
    cvPerfWriter := csv.NewWriter(cvPerfFile)
    cvPerfWriter.Write([]string{
        "Fold", "Accuracy", "Precision", "Recall", "F1",
        "TestSize", "TrainSize", "Correct", "Total",
    })

    folds, _ := cv.KFoldSplit(len(XProcessed), c.loadedData.y)
    cvScores := make([]float64, cvFolds)

    for foldIdx, testIndices := range folds {
        testIndicesMap := make(map[int]bool)
        for _, idx := range testIndices {
            testIndicesMap[idx] = true
        }

        var XTrainCV, XTestCV [][]decimal.Decimal
        var yTrainCV, yTestCV []int

        for j := 0; j < len(XProcessed); j++ {
            if testIndicesMap[j] {
                XTestCV = append(XTestCV, XProcessed[j])
                yTestCV = append(yTestCV, c.loadedData.y[j])
            } else {
                XTrainCV = append(XTrainCV, XProcessed[j])
                yTrainCV = append(yTrainCV, c.loadedData.y[j])
            }
        }

        c.currentModel.Reset()
        c.currentModel.Fit(XTrainCV, yTrainCV)
        foldPredictions := c.currentModel.Predict(XTestCV)

        correct := 0
        for k := range foldPredictions {
            if foldPredictions[k] == yTestCV[k] {
                correct++
            }
        }
        foldAccuracy := float64(correct) / float64(len(yTestCV))
        cvScores[foldIdx] = foldAccuracy

        foldClasses := models.ExtractClasses(yTestCV)
        foldMetrics := evaluation.CalculateMetrics(yTestCV, foldPredictions, foldClasses)

        cvPerfWriter.Write([]string{
            fmt.Sprintf("%d", foldIdx+1),
            fmt.Sprintf("%.4f", foldAccuracy),
            fmt.Sprintf("%.4f", foldMetrics.MacroPrecision),
            fmt.Sprintf("%.4f", foldMetrics.MacroRecall),
            fmt.Sprintf("%.4f", foldMetrics.MacroF1),
            fmt.Sprintf("%d", len(yTestCV)),
            fmt.Sprintf("%d", len(yTrainCV)),
            fmt.Sprintf("%d", correct),
            fmt.Sprintf("%d", len(yTestCV)),
        })
    }
    cvPerfWriter.Flush()

    cvMean := 0.0
    for _, score := range cvScores {
        cvMean += score
    }
    cvMean /= float64(len(cvScores))

    cvVariance := 0.0
    for _, score := range cvScores {
        diff := score - cvMean
        cvVariance += diff * diff
    }
    cvVariance /= float64(len(cvScores))
    cvStd := math.Sqrt(cvVariance)

    c.currentModel.Reset()
    c.currentModel.Fit(XTrainProcessed, yTrain)

    resultsFile, _ := os.Create(filepath.Join(resultsDir, "evaluation_results.csv"))
    defer resultsFile.Close()

    writer := csv.NewWriter(resultsFile)
    writer.Write([]string{
        "Model", "Dataset", "Preprocessing", "TrainSize", "TestSize",
        "Accuracy", "Precision", "Recall", "F1Score",
        "BalancedAccuracy", "CVMean", "CVStd",
    })

    writer.Write([]string{
        modelName,
        datasetName,
        prepMethod,
        strconv.Itoa(len(XTrain)),
        strconv.Itoa(len(XTest)),
        fmt.Sprintf("%.4f", metrics.Accuracy),
        fmt.Sprintf("%.4f", metrics.MacroPrecision),
        fmt.Sprintf("%.4f", metrics.MacroRecall),
        fmt.Sprintf("%.4f", metrics.MacroF1),
        fmt.Sprintf("%.4f", metrics.BalancedAccuracy),
        fmt.Sprintf("%.4f", cvMean),
        fmt.Sprintf("%.4f", cvStd),
    })
    writer.Flush()

    predictionsFile, _ := os.Create(filepath.Join(resultsDir, "predictions.csv"))
    defer predictionsFile.Close()

    predWriter := csv.NewWriter(predictionsFile)
    predWriter.Write([]string{
        "SampleIndex", "ActualClass", "PredictedClass", "Correct", "Confidence",
    })

    for i := range yTest {
        actualClass := ""
        predClass := ""
        if c.loadedData.Classes != nil {
            if yTest[i] < len(c.loadedData.Classes) {
                actualClass = c.loadedData.Classes[yTest[i]]
            }
            if predictions[i] < len(c.loadedData.Classes) {
                predClass = c.loadedData.Classes[predictions[i]]
            }
        } else {
            actualClass = strconv.Itoa(yTest[i])
            predClass = strconv.Itoa(predictions[i])
        }

        confidence := 0.0
        if len(probas) > i {
            classIndex := -1
            for idx, class := range modelClasses {
                if class == predictions[i] {
                    classIndex = idx
                    break
                }
            }
            if classIndex >= 0 && classIndex < len(probas[i]) {
                conf := probas[i][classIndex]
                confidence, _ = conf.Float64()
            }
        }

        predWriter.Write([]string{
            strconv.Itoa(i),
            actualClass,
            predClass,
            strconv.FormatBool(yTest[i] == predictions[i]),
            fmt.Sprintf("%.4f", confidence),
        })
    }
    predWriter.Flush()

    confusionFile, _ := os.Create(filepath.Join(resultsDir, "confusion_matrix.csv"))
    defer confusionFile.Close()

    confWriter := csv.NewWriter(confusionFile)

    header := []string{"Actual\\Predicted"}
    for _, class := range allClasses {
        if c.loadedData.Classes != nil && class < len(c.loadedData.Classes) {
            header = append(header, c.loadedData.Classes[class])
        } else {
            header = append(header, fmt.Sprintf("Class%d", class))
        }
    }
    confWriter.Write(header)

    for _, actualClass := range allClasses {
        row := []string{}
        if c.loadedData.Classes != nil && actualClass < len(c.loadedData.Classes) {
            row = append(row, c.loadedData.Classes[actualClass])
        } else {
            row = append(row, fmt.Sprintf("Class%d", actualClass))
        }

        for _, predClass := range allClasses {
            count := metrics.ConfusionMatrix[actualClass][predClass]
            row = append(row, strconv.Itoa(count))
        }
        confWriter.Write(row)
    }
    confWriter.Flush()

    summaryFile, _ := os.Create(filepath.Join(resultsDir, "summary.txt"))
    defer summaryFile.Close()

    fmt.Fprintf(summaryFile, "Model Evaluation Summary\n")
    fmt.Fprintf(summaryFile, "========================\n\n")
    fmt.Fprintf(summaryFile, "Model: %s\n", modelName)
    fmt.Fprintf(summaryFile, "Dataset: %s\n", c.loadedData.SourceFile)
    fmt.Fprintf(summaryFile, "Timestamp: %s\n\n", timestamp)

    fmt.Fprintf(summaryFile, "Dataset Info:\n")
    fmt.Fprintf(summaryFile, "  Total Samples: %d\n", len(c.loadedData.X))
    fmt.Fprintf(summaryFile, "  Features: %d\n", len(c.loadedData.Features))
    fmt.Fprintf(summaryFile, "  Classes: %d\n\n", len(allClasses))

    fmt.Fprintf(summaryFile, "Performance Metrics:\n")
    fmt.Fprintf(summaryFile, "  Accuracy: %.4f\n", metrics.Accuracy)
    fmt.Fprintf(summaryFile, "  Balanced Accuracy: %.4f\n", metrics.BalancedAccuracy)
    fmt.Fprintf(summaryFile, "  Macro Precision: %.4f\n", metrics.MacroPrecision)
    fmt.Fprintf(summaryFile, "  Macro Recall: %.4f\n", metrics.MacroRecall)
    fmt.Fprintf(summaryFile, "  Macro F1: %.4f\n\n", metrics.MacroF1)

    fmt.Fprintf(summaryFile, "Cross-Validation (%d-fold):\n", cvFolds)
    fmt.Fprintf(summaryFile, "  Mean Accuracy: %.4f\n", cvMean)
    fmt.Fprintf(summaryFile, "  Std Deviation: %.4f\n", cvStd)

    fmt.Printf("%s Results saved to: %s\n", c.green("✓"), resultsDir)
    fmt.Printf("  • evaluation_results.csv - Main metrics\n")
    fmt.Printf("  • predictions.csv - Detailed predictions\n")
    fmt.Printf("  • cv_performance.csv - Per-fold CV metrics\n")
    fmt.Printf("  • confusion_matrix.csv - Confusion matrix\n")
    fmt.Printf("  • summary.txt - Human-readable summary\n")
}

func (c *Commander) quit() {
    os.Exit(0)
}