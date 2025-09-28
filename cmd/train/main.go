package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"mlclassifier/internal/data"
	"mlclassifier/internal/evaluation"
	"mlclassifier/internal/experiment"
	"mlclassifier/internal/models"
	"mlclassifier/internal/persistence"
	"mlclassifier/internal/preprocessing"
)

func main() {
	dataFile := flag.String("data", "", "Path to training data CSV file")
	algorithm := flag.String("algorithm", "knn", "Algorithm to use (knn|tree|forest|bayes)")
	configFile := flag.String("config", "config/config.yaml", "Path to configuration file")
	outputDir := flag.String("output", "models", "Output directory for trained models")
	preprocessing := flag.String("preprocess", "normalized", "Preprocessing method (raw|normalized|standardized)")
	experiment := flag.Bool("experiment", false, "Run full experiment with config")
	k := flag.Int("k", 5, "K value for KNN")
	maxDepth := flag.Int("max-depth", 10, "Max depth for decision tree/forest")
	nTrees := flag.Int("n-trees", 100, "Number of trees for random forest")
	testSize := flag.Float64("test-size", 0.2, "Test set size (0.0-1.0)")
	crossValidation := flag.Bool("cv", true, "Enable cross-validation")
	cvFolds := flag.Int("cv-folds", 5, "Number of cross-validation folds")

	flag.Parse()

	if *dataFile == "" && !*experiment {
		fmt.Println("Usage:")
		fmt.Println("  Simple training: go run cmd/train/main.go -data data/train/iris.csv -algorithm knn")
		fmt.Println("  Full experiment: go run cmd/train/main.go -experiment -config config/config.yaml -data data/train/iris.csv")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	if *experiment {
		runExperiment(*configFile, *dataFile, *outputDir)
	} else {
		runSingleTraining(*dataFile, *algorithm, *preprocessing, *outputDir, *k, *maxDepth, *nTrees, *testSize, *crossValidation, *cvFolds)
	}
}

func runExperiment(configFile, dataFile, outputDir string) {
	fmt.Println("Running full experiment...")

	runner := experiment.NewRunner(configFile)
	results, err := runner.RunAllExperiments(dataFile)
	if err != nil {
		log.Fatalf("Experiment failed: %v", err)
	}

	timestamp := time.Now().Format("20060102_150405")
	expDir := filepath.Join(outputDir, fmt.Sprintf("experiment_%s", timestamp))
	os.MkdirAll(expDir, 0755)

	resultsFile := filepath.Join(expDir, "experiment_results.csv")
	if err := runner.ExportResults(results, resultsFile); err != nil {
		log.Printf("Failed to export results: %v", err)
	} else {
		fmt.Printf("Experiment results saved to: %s\n", resultsFile)
	}

	fmt.Printf("\nExperiment Summary:\n")
	fmt.Printf("Total experiments: %d\n", len(results))

	if len(results) > 0 {
		best := results[0]
		for _, result := range results[1:] {
			if result.Accuracy > best.Accuracy {
				best = result
			}
		}
		fmt.Printf("Best accuracy: %.4f (%s with %s preprocessing)\n",
			best.Accuracy, best.Algorithm, best.Preprocessing)
	}
}

func runSingleTraining(dataFile, algorithm, preprocessMethod, outputDir string, k, maxDepth, nTrees int, testSize float64, crossValidation bool, cvFolds int) {
	fmt.Printf("Training %s model on %s...\n", algorithm, dataFile)

	fmt.Println("Loading dataset...")
	reader, err := data.NewCSVReader(dataFile)
	if err != nil {
		log.Fatalf("Failed to create CSV reader: %v", err)
	}

	X, y, headers, err := reader.LoadData()
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}

	fmt.Printf("Loaded %d samples with %d features\n", len(X), len(headers))

	validator := data.NewDataValidator()
	if err := validator.ValidateDataset(X, y); err != nil {
		log.Fatalf("Data validation failed: %v", err)
	}

	fmt.Printf("Applying %s preprocessing...\n", preprocessMethod)
	var XProcessed = X
	if preprocessMethod != "raw" {
		scaler := preprocessing.NewScaler(preprocessMethod)
		XProcessed, err = scaler.FitTransform(X)
		if err != nil {
			log.Fatalf("Preprocessing failed: %v", err)
		}
	}

	fmt.Printf("Splitting data (test size: %.1f%%)...\n", testSize*100)
	splitter := evaluation.NewTrainTestSplitter(testSize, time.Now().UnixNano(), true)
	XTrain, XTest, yTrain, yTest, err := splitter.StratifiedSplit(XProcessed, y)
	if err != nil {
		log.Fatalf("Failed to split data: %v", err)
	}

	config := models.ModelConfig{
		Algorithm:    algorithm,
		K:            k,
		Distance:     "euclidean",
		MaxDepth:     maxDepth,
		MinSplit:     2,
		NTrees:       nTrees,
		VarSmoothing: 1e-9,
	}

	model, err := models.CreateModel(config)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	fmt.Printf("Training %s model...\n", model.GetName())
	startTime := time.Now()
	if err := model.Fit(XTrain, yTrain); err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	trainingTime := time.Since(startTime)

	fmt.Println("Evaluating model...")
	predictions := model.Predict(XTest)
	classes := models.ExtractClasses(y)
	metrics := evaluation.CalculateMetrics(yTest, predictions, classes)

	fmt.Printf("\nTraining Results:\n")
	fmt.Printf("Training time: %v\n", trainingTime)
	fmt.Printf("Test accuracy: %.4f\n", metrics.Accuracy)
	fmt.Printf("Precision: %.4f\n", metrics.MacroPrecision)
	fmt.Printf("Recall: %.4f\n", metrics.MacroRecall)
	fmt.Printf("F1-score: %.4f\n", metrics.MacroF1)

	if crossValidation {
		fmt.Printf("Running %d-fold cross-validation...\n", cvFolds)
		cv := evaluation.NewCrossValidator(cvFolds, true)
		_, mean, std, err := cv.ParallelCrossValidate(XProcessed, y, model)
		if err != nil {
			fmt.Printf("Parallel CV failed, trying serial: %v\n", err)
			_, mean, std, _ = cv.CrossValidateSerial(XProcessed, y, model)
		}
		fmt.Printf("CV accuracy: %.4f ± %.4f\n", mean, std)
	}

	fmt.Println("Saving model...")
	os.MkdirAll(outputDir, 0755)
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("%s_%s_%s_%s.model", algorithm,
		filepath.Base(dataFile[:len(dataFile)-4]), preprocessMethod, timestamp)
	modelPath := filepath.Join(outputDir, filename)

	bundle := persistence.NewModelBundle(model)
	bundle.Metadata.Dataset = dataFile
	bundle.Metadata.Accuracy = metrics.Accuracy
	bundle.Metadata.Precision = metrics.MacroPrecision
	bundle.Metadata.Recall = metrics.MacroRecall
	bundle.Metadata.F1Score = metrics.MacroF1
	bundle.Metadata.TrainingTime = trainingTime

	if err := bundle.Save(modelPath); err != nil {
		log.Printf("Failed to save model: %v", err)
	} else {
		fmt.Printf("Model saved to: %s\n", modelPath)
	}

	fmt.Println("\nTraining completed successfully!")
}
