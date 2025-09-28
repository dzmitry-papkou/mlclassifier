package commander

import (
    "context"
    "encoding/csv"
    "fmt"
    "io"
    "os"
    "path/filepath"
    "sort"
    "strconv"
    "strings"
    "time"

    "mlclassifier/internal/data"
    "mlclassifier/internal/evaluation"
    "mlclassifier/internal/jobs"
    "mlclassifier/internal/models"
    "mlclassifier/internal/persistence"
    "mlclassifier/internal/preprocessing"

    "github.com/shopspring/decimal"
)

func (c *Commander) loadStreamingData(args []string) {
    if len(args) == 0 {
        fmt.Println(c.red("Please specify a file to load"))
        return
    }

    filename := args[0]
    batchSize := 1000

    for _, arg := range args[1:] {
        if value, ok := strings.CutPrefix(arg, "--batch-size="); ok {
            val, err := strconv.Atoi(value)
            if err == nil && val > 0 {
                batchSize = val
            }
        }
    }

    fmt.Printf("Loading %s with batch size %d...\n", filename, batchSize)

    streamer, err := data.NewStreamingReader(filename, batchSize, -1)
    if err != nil {
        fmt.Printf("%s Failed to open file: %v\n", c.red("✗"), err)
        return
    }
    defer streamer.Close()

    var allX [][]decimal.Decimal
    var allY []int
    var headers []string
    batchNum := 0

    for {
        batch, err := streamer.ReadBatch()
        if err != nil {
            if err == io.EOF {
                break
            }
            fmt.Printf("%s Error reading batch: %v\n", c.red("✗"), err)
            break
        }

        if batchNum == 0 {
            headers = streamer.GetHeaders()
        }

        encodedLabels, _ := streamer.GetEncoder().Transform(batch.Labels)

        allX = append(allX, batch.X...)
        allY = append(allY, encodedLabels...)
        batchNum++

        fmt.Printf("\rLoaded %d batches (%d samples)...", batchNum, len(allX))
    }

    fmt.Printf("\n%s Successfully loaded %d samples\n", c.green("✓"), len(allX))

    classes := models.ExtractClasses(allY)
    classNames := make([]string, len(classes))
    for i, cls := range classes {
        classNames[i] = fmt.Sprintf("Class_%d", cls)
    }

    c.loadedData = &DataSet{
        X:          allX,
        y:          allY,
        Features:   headers,
        Classes:    classNames,
        SourceFile: filename,
    }

    c.showDataInfo()
}

func (c *Commander) trainModelBackground(algorithm string, params []string) {
    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded. Use 'load <file>' first"))
        return
    }

    job := c.jobManager.CreateJob("train", fmt.Sprintf("Training %s model", algorithm))
    fmt.Printf("Job submitted: %s\n", c.cyan(job.ID))

    go func() {
        ctx, cancel := context.WithCancel(context.Background())
        job.SetCancelFunc(cancel)
        job.SetStatus(jobs.JobRunning)
        job.AddLog(fmt.Sprintf("Starting training of %s model", algorithm))

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
        }

        model, err := models.CreateModel(config)
        if err != nil {
            job.SetError(err)
            return
        }

        job.AddLog("Preprocessing data...")
        job.SetProgress(0.2)

        splitter := evaluation.DefaultTrainTestSplitter()
        XTrain, XTest, yTrain, yTest, err := splitter.StratifiedSplit(c.loadedData.X, c.loadedData.y)
        if err != nil {
            job.SetError(err)
            return
        }

        job.AddLog(fmt.Sprintf("Training with %d samples...", len(XTrain)))
        job.SetProgress(0.4)

        select {
        case <-ctx.Done():
            job.SetStatus(jobs.JobCancelled)
            job.AddLog("Training cancelled by user")
            return
        default:
        }

        if err := model.Fit(XTrain, yTrain); err != nil {
            job.SetError(err)
            return
        }

        job.AddLog("Evaluating model...")
        job.SetProgress(0.8)

        predictions := model.Predict(XTest)
        classes := models.ExtractClasses(c.loadedData.y)
        metrics := evaluation.CalculateMetrics(yTest, predictions, classes)

        job.SetProgress(1.0)
        job.SetResult(metrics)
        job.SetStatus(jobs.JobCompleted)
        job.AddLog(fmt.Sprintf("Training completed. Accuracy: %.4f", metrics.Accuracy))

        timestamp := time.Now().Format("20060102_150405")
        filename := fmt.Sprintf("models/%s_%s_bg_%s.model",
            algorithm, filepath.Base(c.loadedData.SourceFile[:len(c.loadedData.SourceFile)-4]), timestamp)

        bundle := persistence.NewModelBundle(model)
        bundle.Metadata.Dataset = c.loadedData.SourceFile
        bundle.Metadata.Accuracy = metrics.Accuracy

        if err := bundle.Save(filename); err != nil {
            job.AddLog(fmt.Sprintf("Failed to save model: %v", err))
        } else {
            job.AddLog(fmt.Sprintf("Model saved to: %s", filename))
        }
    }()
}

func (c *Commander) crossValidateAdvanced() {
    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded. Use 'load <file>' first"))
        return
    }

    fmt.Println(c.cyan("Advanced Cross-Validation"))
    fmt.Println("Select algorithms (comma-separated, or '5' for all):")
    fmt.Println("1. KNN")
    fmt.Println("2. Decision Tree")
    fmt.Println("3. Random Forest")
    fmt.Println("4. Naive Bayes")
    fmt.Println("5. All")

    var selection string
    fmt.Print("Selection: ")
    fmt.Scanln(&selection)

    var selectedAlgos []string
    if selection == "5" {
        selectedAlgos = []string{"knn", "tree", "forest", "bayes"}
    } else {
        parts := strings.Split(selection, ",")
        for _, part := range parts {
            switch strings.TrimSpace(part) {
            case "1":
                selectedAlgos = append(selectedAlgos, "knn")
            case "2":
                selectedAlgos = append(selectedAlgos, "tree")
            case "3":
                selectedAlgos = append(selectedAlgos, "forest")
            case "4":
                selectedAlgos = append(selectedAlgos, "bayes")
            }
        }
    }

    if len(selectedAlgos) == 0 {
        fmt.Println(c.red("No algorithms selected"))
        return
    }

    var folds int
    fmt.Print("Number of folds: ")
    fmt.Scanln(&folds)
    if folds < 2 {
        folds = 10
    }

    var repetitions int
    fmt.Print("Repetitions: ")
    fmt.Scanln(&repetitions)
    if repetitions < 1 {
        repetitions = 3
    }

    fmt.Printf("\nRunning %d-fold CV with %d repetitions on %d algorithms...\n",
        folds, repetitions, len(selectedAlgos))

    results := make(map[string][]float64)
    cv := evaluation.NewCrossValidator(folds, true)

    for _, algo := range selectedAlgos {
        fmt.Printf("\nEvaluating %s...\n", algo)
        config := models.DefaultConfig(algo)

        var allScores []float64
        for rep := 0; rep < repetitions; rep++ {
            model, _ := models.CreateModel(config)
            scores, mean, std, err := cv.ParallelCrossValidate(c.loadedData.X, c.loadedData.y, model)
            if err != nil {
                fmt.Printf("  Rep %d failed: %v\n", rep+1, err)
                continue
            }
            allScores = append(allScores, scores...)
            fmt.Printf("  Rep %d: %.4f ± %.4f\n", rep+1, mean, std)
        }
        results[algo] = allScores
    }

    timestamp := time.Now().Format("20060102_150405")
    filename := fmt.Sprintf("results/cv_advanced_%s.csv", timestamp)
    c.exportCVResults(filename, results, folds, repetitions)
    fmt.Printf("\n%s Results exported to %s\n", c.green("✓"), filename)
}

func (c *Commander) runExperimentBackground() {
    if c.loadedData == nil {
        fmt.Println(c.red("No data loaded. Use 'load <file>' first"))
        return
    }

    job := c.jobManager.CreateJob("experiment", "Running comprehensive experiment")
    fmt.Printf("Job submitted: %s\n", c.cyan(job.ID))
    fmt.Println("Use 'job-status' to monitor progress")

    go func() {
        ctx, cancel := context.WithCancel(context.Background())
        job.SetCancelFunc(cancel)
        job.SetStatus(jobs.JobRunning)
        job.AddLog("Starting comprehensive experiment")

        timestamp := time.Now().Format("20060102_150405")
        expDir := fmt.Sprintf("results/exp_bg_%s", timestamp)
        os.MkdirAll(expDir, 0755)

        algorithms := []models.ModelConfig{
            {Algorithm: "knn", K: 5, Distance: "euclidean"},
            {Algorithm: "tree", MaxDepth: 10, MinSplit: 2},
            {Algorithm: "forest", NTrees: 50, MaxDepth: 10, MinSplit: 2},
            {Algorithm: "bayes", VarSmoothing: 1e-9},
        }

        preprocessingMethods := []string{"raw", "normalized", "standardized"}

        totalTasks := len(algorithms) * len(preprocessingMethods)
        completedTasks := 0

        type experimentResult struct {
            Algorithm     string
            Preprocessing string
            Accuracy      float64
            TrainingTime  time.Duration
            Error         error
        }

        results := []experimentResult{}

        for _, prepMethod := range preprocessingMethods {
            for _, algoConfig := range algorithms {
                select {
                case <-ctx.Done():
                    job.SetStatus(jobs.JobCancelled)
                    job.AddLog("Experiment cancelled by user")
                    return
                default:
                }

                job.AddLog(fmt.Sprintf("Testing %s with %s preprocessing...",
                    algoConfig.Algorithm, prepMethod))

                model, err := models.CreateModel(algoConfig)
                if err != nil {
                    job.AddLog(fmt.Sprintf("Failed to create %s: %v", algoConfig.Algorithm, err))
                    continue
                }

                XProcessed := c.loadedData.X
                if prepMethod != "raw" {
                    scaler := preprocessing.NewScaler(prepMethod)
                    XProcessed, _ = scaler.FitTransform(c.loadedData.X)
                }

                splitter := evaluation.DefaultTrainTestSplitter()
                XTrain, XTest, yTrain, yTest, _ := splitter.StratifiedSplit(XProcessed, c.loadedData.y)

                startTime := time.Now()
                err = model.Fit(XTrain, yTrain)
                trainingTime := time.Since(startTime)

                if err != nil {
                    results = append(results, experimentResult{
                        Algorithm:     algoConfig.Algorithm,
                        Preprocessing: prepMethod,
                        Error:         err,
                    })
                    job.AddLog(fmt.Sprintf("Training failed: %v", err))
                } else {
                    predictions := model.Predict(XTest)
                    classes := models.ExtractClasses(c.loadedData.y)
                    metrics := evaluation.CalculateMetrics(yTest, predictions, classes)

                    results = append(results, experimentResult{
                        Algorithm:     algoConfig.Algorithm,
                        Preprocessing: prepMethod,
                        Accuracy:      metrics.Accuracy,
                        TrainingTime:  trainingTime,
                    })

                    job.AddLog(fmt.Sprintf("  Accuracy: %.4f", metrics.Accuracy))
                }

                completedTasks++
                job.SetProgress(float64(completedTasks) / float64(totalTasks))
            }
        }

        resultsFile := filepath.Join(expDir, "experiment_results.csv")
        file, err := os.Create(resultsFile)
        if err == nil {
            writer := csv.NewWriter(file)
            writer.Write([]string{"Algorithm", "Preprocessing", "Accuracy", "TrainingTime"})

            for _, r := range results {
                if r.Error == nil {
                    writer.Write([]string{
                        r.Algorithm,
                        r.Preprocessing,
                        fmt.Sprintf("%.4f", r.Accuracy),
                        r.TrainingTime.String(),
                    })
                }
            }
            writer.Flush()
            file.Close()
            job.AddLog(fmt.Sprintf("Results saved to: %s", resultsFile))
        }

        var bestResult experimentResult
        bestAccuracy := 0.0
        for _, r := range results {
            if r.Error == nil && r.Accuracy > bestAccuracy {
                bestAccuracy = r.Accuracy
                bestResult = r
            }
        }

        job.SetResult(bestResult)
        job.SetStatus(jobs.JobCompleted)
        job.AddLog(fmt.Sprintf("Experiment completed. Best: %s with %s (%.4f)",
            bestResult.Algorithm, bestResult.Preprocessing, bestResult.Accuracy))
    }()
}

func (c *Commander) listAllJobs() {
    jobs := c.jobManager.ListJobs()
    if len(jobs) == 0 {
        fmt.Println("No jobs found")
        return
    }

    fmt.Println(c.cyan("Background Jobs:"))
    fmt.Println(strings.Repeat("-", 80))
    fmt.Printf("%-20s %-10s %-10s %-15s %s\n", "Job ID", "Type", "Status", "Progress", "Description")
    fmt.Println(strings.Repeat("-", 80))

    for _, job := range jobs {
        statusColor := c.yellow
        switch job.GetStatus() {
        case "completed":
            statusColor = c.green
        case "failed":
            statusColor = c.red
        case "running":
            statusColor = c.cyan
        }

        progress := fmt.Sprintf("%.0f%%", job.GetProgress()*100)
        fmt.Printf("%-20s %-10s %-10s %-15s %s\n",
            job.ID, job.Type, statusColor(string(job.GetStatus())), progress, job.Description)
    }
}

func (c *Commander) showJobStatus(jobID string) {
    job, exists := c.jobManager.GetJob(jobID)
    if !exists {
        fmt.Printf("%s Job not found: %s\n", c.red("✗"), jobID)
        return
    }

    fmt.Printf("\n%s\n", c.cyan("Job Details:"))
    fmt.Printf("ID:          %s\n", job.ID)
    fmt.Printf("Type:        %s\n", job.Type)
    fmt.Printf("Status:      %s\n", job.GetStatus())
    fmt.Printf("Progress:    %.0f%%\n", job.GetProgress()*100)
    fmt.Printf("Start Time:  %s\n", job.StartTime.Format("15:04:05"))
    if job.EndTime != nil {
        fmt.Printf("End Time:    %s\n", job.EndTime.Format("15:04:05"))
        fmt.Printf("Duration:    %s\n", job.EndTime.Sub(job.StartTime))
    }
    if job.Error != nil {
        fmt.Printf("Error:       %s\n", c.red(job.Error.Error()))
    }
}

func (c *Commander) cancelJob(jobID string) {
    err := c.jobManager.CancelJob(jobID)
    if err != nil {
        fmt.Printf("%s %v\n", c.red("✗"), err)
    } else {
        fmt.Printf("%s Job cancelled: %s\n", c.green("✓"), jobID)
    }
}

func (c *Commander) showJobLogs(jobID string) {
    job, exists := c.jobManager.GetJob(jobID)
    if !exists {
        fmt.Printf("%s Job not found: %s\n", c.red("✗"), jobID)
        return
    }

    logs := job.GetLogs()
    if len(logs) == 0 {
        fmt.Println("No logs available")
        return
    }

    fmt.Printf("\n%s\n", c.cyan(fmt.Sprintf("Logs for job %s:", jobID)))
    for _, log := range logs {
        fmt.Println(log)
    }
}

func (c *Commander) listModelVersions() {
    modelDir := "models"
    files, err := os.ReadDir(modelDir)
    if err != nil {
        fmt.Printf("%s Failed to read models directory: %v\n", c.red("✗"), err)
        return
    }

    type modelVersion struct {
        filename  string
        algorithm string
        dataset   string
        timestamp time.Time
        size      int64
    }

    var versions []modelVersion
    for _, entry := range files {
        if strings.HasSuffix(entry.Name(), ".model") || strings.HasSuffix(entry.Name(), ".bundle") {
            info, _ := entry.Info()
            parts := strings.Split(entry.Name(), "_")
            if len(parts) >= 3 {
                v := modelVersion{
                    filename:  entry.Name(),
                    algorithm: parts[0],
                    dataset:   parts[1],
                    size:      info.Size(),
                }

                timeStr := strings.TrimSuffix(parts[len(parts)-1], ".model")
                timeStr = strings.TrimSuffix(timeStr, ".bundle")
                if t, err := time.Parse("20060102", timeStr[:8]); err == nil {
                    v.timestamp = t
                } else {
                    v.timestamp = info.ModTime()
                }

                versions = append(versions, v)
            }
        }
    }

    if len(versions) == 0 {
        fmt.Println("No model versions found")
        return
    }

    sort.Slice(versions, func(i, j int) bool {
        return versions[i].timestamp.After(versions[j].timestamp)
    })

    fmt.Println(c.cyan("Model Versions:"))
    fmt.Println(strings.Repeat("-", 80))
    fmt.Printf("%-30s %-10s %-15s %-10s %s\n", "Filename", "Algorithm", "Dataset", "Size", "Date")
    fmt.Println(strings.Repeat("-", 80))

    for _, v := range versions {
        sizeKB := float64(v.size) / 1024
        fmt.Printf("%-30s %-10s %-15s %-10.2fKB %s\n",
            v.filename, v.algorithm, v.dataset, sizeKB, v.timestamp.Format("2006-01-02"))
    }
}

func (c *Commander) compareModelVersions(version1, version2 string) {
    fmt.Printf("Comparing %s vs %s...\n", c.cyan(version1), c.cyan(version2))

    if !strings.Contains(version1, "/") {
        version1 = filepath.Join("models", version1)
    }
    if !strings.Contains(version2, "/") {
        version2 = filepath.Join("models", version2)
    }

    bundle1, err := persistence.LoadModelBundle(version1)
    if err != nil {
        fmt.Printf("%s Failed to load %s: %v\n", c.red("✗"), version1, err)
        return
    }

    bundle2, err := persistence.LoadModelBundle(version2)
    if err != nil {
        fmt.Printf("%s Failed to load %s: %v\n", c.red("✗"), version2, err)
        return
    }

    fmt.Println(c.cyan("\n╔══════════════════════════════════════════════════════╗"))
    fmt.Println(c.cyan("║                 Model Comparison                      ║"))
    fmt.Println(c.cyan("╚══════════════════════════════════════════════════════╝"))

    fmt.Printf("\n%-20s %-25s %-25s\n", "Metric", version1, version2)
    fmt.Println(strings.Repeat("-", 70))

    fmt.Printf("%-20s %-25s %-25s\n", "Algorithm:", bundle1.Model.GetName(), bundle2.Model.GetName())
    fmt.Printf("%-20s %-25s %-25s\n", "Dataset:", bundle1.Metadata.Dataset, bundle2.Metadata.Dataset)

    compareMetric := func(name string, val1, val2 float64) {
        v1Str := fmt.Sprintf("%.4f", val1)
        v2Str := fmt.Sprintf("%.4f", val2)

        if val1 > val2 {
            v1Str = c.green(v1Str + " ▲")
        } else if val2 > val1 {
            v2Str = c.green(v2Str + " ▲")
        }

        fmt.Printf("%-20s %-25s %-25s\n", name, v1Str, v2Str)
    }

    compareMetric("Accuracy:", bundle1.Metadata.Accuracy, bundle2.Metadata.Accuracy)
    compareMetric("Precision:", bundle1.Metadata.Precision, bundle2.Metadata.Precision)
    compareMetric("Recall:", bundle1.Metadata.Recall, bundle2.Metadata.Recall)
    compareMetric("F1-Score:", bundle1.Metadata.F1Score, bundle2.Metadata.F1Score)

    t1Str := bundle1.Metadata.TrainingTime.String()
    t2Str := bundle2.Metadata.TrainingTime.String()
    if bundle1.Metadata.TrainingTime < bundle2.Metadata.TrainingTime {
        t1Str = c.green(t1Str + " ⚡")
    } else if bundle2.Metadata.TrainingTime < bundle1.Metadata.TrainingTime {
        t2Str = c.green(t2Str + " ⚡")
    }
    fmt.Printf("%-20s %-25s %-25s\n", "Training Time:", t1Str, t2Str)

    fmt.Println(c.cyan("\n" + strings.Repeat("-", 70)))
    if bundle1.Metadata.Accuracy > bundle2.Metadata.Accuracy {
        fmt.Printf("%s %s performs better overall\n", c.green("Recommendation:"), filepath.Base(version1))
    } else if bundle2.Metadata.Accuracy > bundle1.Metadata.Accuracy {
        fmt.Printf("%s %s performs better overall\n", c.green("Recommendation:"), filepath.Base(version2))
    } else {
        fmt.Printf("%s Both models have similar performance\n", c.yellow("Note:"))
    }
}

func (c *Commander) promoteModel(version string) {
    fmt.Printf("Promoting model %s to production...\n", c.cyan(version))

    if !strings.Contains(version, "/") {
        version = filepath.Join("models", version)
    }

    if _, err := os.Stat(version); os.IsNotExist(err) {
        fmt.Printf("%s Model not found: %s\n", c.red("✗"), version)
        return
    }

    prodPath := filepath.Join("models", "production.model")
    if _, err := os.Stat(prodPath); err == nil {
        backupPath := filepath.Join("models", fmt.Sprintf("production_backup_%s.model",
            time.Now().Format("20060102_150405")))

        data, err := os.ReadFile(prodPath)
        if err == nil {
            os.WriteFile(backupPath, data, 0644)
            fmt.Printf("%s Previous production model backed up to: %s\n",
                c.yellow("ℹ"), filepath.Base(backupPath))
        }
    }

    sourceData, err := os.ReadFile(version)
    if err != nil {
        fmt.Printf("%s Failed to read model: %v\n", c.red("✗"), err)
        return
    }

    err = os.WriteFile(prodPath, sourceData, 0644)
    if err != nil {
        fmt.Printf("%s Failed to promote model: %v\n", c.red("✗"), err)
        return
    }

    logFile := filepath.Join("models", "promotion_log.txt")
    logEntry := fmt.Sprintf("[%s] Promoted %s to production\n",
        time.Now().Format("2006-01-02 15:04:05"), filepath.Base(version))

    f, _ := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if f != nil {
        f.WriteString(logEntry)
        f.Close()
    }

    fmt.Printf("%s Model promoted successfully to production\n", c.green("✓"))
    fmt.Printf("  Production model: %s\n", prodPath)
}

func (c *Commander) rollbackModel(version string) {
    fmt.Printf("Rolling back to model %s...\n", c.cyan(version))

    if version == "last" {
        modelDir := "models"
        entries, err := os.ReadDir(modelDir)
        if err != nil {
            fmt.Printf("%s Failed to read models directory: %v\n", c.red("✗"), err)
            return
        }

        var latestBackup string
        var latestTime time.Time

        for _, entry := range entries {
            if strings.HasPrefix(entry.Name(), "production_backup_") {
                info, _ := entry.Info()
                if info.ModTime().After(latestTime) {
                    latestTime = info.ModTime()
                    latestBackup = entry.Name()
                }
            }
        }

        if latestBackup == "" {
            fmt.Printf("%s No backup found to rollback to\n", c.red("✗"))
            return
        }
        version = latestBackup
    }

    if !strings.Contains(version, "/") {
        version = filepath.Join("models", version)
    }

    if _, err := os.Stat(version); os.IsNotExist(err) {
        fmt.Printf("%s Rollback model not found: %s\n", c.red("✗"), version)
        return
    }

    prodPath := filepath.Join("models", "production.model")
    if _, err := os.Stat(prodPath); err == nil {
        backupPath := filepath.Join("models", fmt.Sprintf("production_before_rollback_%s.model",
            time.Now().Format("20060102_150405")))

        data, err := os.ReadFile(prodPath)
        if err == nil {
            os.WriteFile(backupPath, data, 0644)
            fmt.Printf("%s Current model backed up to: %s\n",
                c.yellow("ℹ"), filepath.Base(backupPath))
        }
    }

    rollbackData, err := os.ReadFile(version)
    if err != nil {
        fmt.Printf("%s Failed to read rollback model: %v\n", c.red("✗"), err)
        return
    }

    err = os.WriteFile(prodPath, rollbackData, 0644)
    if err != nil {
        fmt.Printf("%s Failed to rollback model: %v\n", c.red("✗"), err)
        return
    }

    logFile := filepath.Join("models", "promotion_log.txt")
    logEntry := fmt.Sprintf("[%s] Rolled back to %s\n",
        time.Now().Format("2006-01-02 15:04:05"), filepath.Base(version))

    f, _ := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if f != nil {
        f.WriteString(logEntry)
        f.Close()
    }

    fmt.Printf("%s Model rolled back successfully\n", c.green("✓"))
    fmt.Printf("  Production model restored from: %s\n", filepath.Base(version))
}

func (c *Commander) listExperiments() {
    resultsDir := "results"
    entries, err := os.ReadDir(resultsDir)
    if err != nil {
        fmt.Printf("%s Failed to read results directory: %v\n", c.red("✗"), err)
        return
    }

    type experiment struct {
        name    string
        modTime time.Time
    }

    experiments := []experiment{}
    for _, entry := range entries {
        if entry.IsDir() && strings.HasPrefix(entry.Name(), "exp_") {
            info, _ := entry.Info()
            experiments = append(experiments, experiment{
                name:    entry.Name(),
                modTime: info.ModTime(),
            })
        }
    }

    if len(experiments) == 0 {
        fmt.Println("No experiments found")
        return
    }

    fmt.Println(c.cyan("Past Experiments:"))
    fmt.Println(strings.Repeat("-", 60))
    fmt.Printf("%-40s %s\n", "Experiment ID", "Date")
    fmt.Println(strings.Repeat("-", 60))

    for _, exp := range experiments {
        fmt.Printf("%-40s %s\n", exp.name, exp.modTime.Format("2006-01-02 15:04"))
    }
}

func (c *Commander) viewExperiment(expID string) {
    expDir := filepath.Join("results", expID)
    summaryFile := filepath.Join(expDir, "experiment_summary.txt")

    content, err := os.ReadFile(summaryFile)
    if err != nil {
        fmt.Printf("%s Failed to read experiment: %v\n", c.red("✗"), err)
        return
    }

    fmt.Printf("\n%s\n", c.cyan(fmt.Sprintf("Experiment: %s", expID)))
    fmt.Println(string(content))
}

func (c *Commander) exportCVResults(filename string, results map[string][]float64, folds, reps int) {
    os.MkdirAll(filepath.Dir(filename), 0755)
    file, err := os.Create(filename)
    if err != nil {
        fmt.Printf("%s Failed to create file: %v\n", c.red("✗"), err)
        return
    }
    defer file.Close()

    writer := csv.NewWriter(file)
    defer writer.Flush()

    writer.Write([]string{"Algorithm", "Folds", "Repetitions", "Mean", "StdDev", "Min", "Max"})

    for algo, scores := range results {
        if len(scores) == 0 {
            continue
        }

        mean := 0.0
        for _, s := range scores {
            mean += s
        }
        mean /= float64(len(scores))

        variance := 0.0
        for _, s := range scores {
            variance += (s - mean) * (s - mean)
        }
        stddev := 0.0
        if len(scores) > 1 {
            stddev = variance / float64(len(scores)-1)
        }

        min, max := scores[0], scores[0]
        for _, s := range scores {
            if s < min {
                min = s
            }
            if s > max {
                max = s
            }
        }

        writer.Write([]string{
            algo,
            strconv.Itoa(folds),
            strconv.Itoa(reps),
            fmt.Sprintf("%.4f", mean),
            fmt.Sprintf("%.4f", stddev),
            fmt.Sprintf("%.4f", min),
            fmt.Sprintf("%.4f", max),
        })
    }
}