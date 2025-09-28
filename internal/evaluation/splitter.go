package evaluation

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/shopspring/decimal"
)

type TrainTestSplitter struct {
	testSize   float64
	randomSeed int64
	shuffle    bool
}

func NewTrainTestSplitter(testSize float64, randomSeed int64, shuffle bool) *TrainTestSplitter {
	return &TrainTestSplitter{
		testSize:   testSize,
		randomSeed: randomSeed,
		shuffle:    shuffle,
	}
}

func (tts *TrainTestSplitter) Split(X [][]decimal.Decimal, y []int) ([][]decimal.Decimal, [][]decimal.Decimal, []int, []int, error) {
	if len(X) != len(y) {
		return nil, nil, nil, nil, fmt.Errorf("x and y must have the same length")
	}

	if len(X) == 0 {
		return nil, nil, nil, nil, fmt.Errorf("cannot split empty dataset")
	}

	if tts.testSize <= 0 || tts.testSize >= 1 {
		return nil, nil, nil, nil, fmt.Errorf("test size must be between 0 and 1")
	}

	n := len(X)
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	if tts.shuffle {
		rng := rand.New(rand.NewSource(tts.randomSeed))
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	testCount := int(float64(n) * tts.testSize)
	trainCount := n - testCount

	XTrain := make([][]decimal.Decimal, trainCount)
	XTest := make([][]decimal.Decimal, testCount)
	yTrain := make([]int, trainCount)
	yTest := make([]int, testCount)

	for i := 0; i < trainCount; i++ {
		idx := indices[i]
		XTrain[i] = make([]decimal.Decimal, len(X[idx]))
		copy(XTrain[i], X[idx])
		yTrain[i] = y[idx]
	}

	for i := 0; i < testCount; i++ {
		idx := indices[trainCount+i]
		XTest[i] = make([]decimal.Decimal, len(X[idx]))
		copy(XTest[i], X[idx])
		yTest[i] = y[idx]
	}

	return XTrain, XTest, yTrain, yTest, nil
}

func (tts *TrainTestSplitter) StratifiedSplit(X [][]decimal.Decimal, y []int) ([][]decimal.Decimal, [][]decimal.Decimal, []int, []int, error) {
	if len(X) != len(y) {
		return nil, nil, nil, nil, fmt.Errorf("x and y must have the same length")
	}

	if len(X) == 0 {
		return nil, nil, nil, nil, fmt.Errorf("cannot split empty dataset")
	}

	classIndices := make(map[int][]int)
	for i, label := range y {
		classIndices[label] = append(classIndices[label], i)
	}

	var trainIndices, testIndices []int

	rng := rand.New(rand.NewSource(tts.randomSeed))
	for _, indices := range classIndices {
		if tts.shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		testCount := int(float64(len(indices)) * tts.testSize)
		if testCount == 0 && len(indices) > 0 {
			testCount = 1
		}

		trainCount := len(indices) - testCount

		for i := 0; i < trainCount; i++ {
			trainIndices = append(trainIndices, indices[i])
		}
		for i := trainCount; i < len(indices); i++ {
			testIndices = append(testIndices, indices[i])
		}
	}

	if tts.shuffle {
		rng.Shuffle(len(trainIndices), func(i, j int) {
			trainIndices[i], trainIndices[j] = trainIndices[j], trainIndices[i]
		})
		rng.Shuffle(len(testIndices), func(i, j int) {
			testIndices[i], testIndices[j] = testIndices[j], testIndices[i]
		})
	}

	XTrain := make([][]decimal.Decimal, len(trainIndices))
	XTest := make([][]decimal.Decimal, len(testIndices))
	yTrain := make([]int, len(trainIndices))
	yTest := make([]int, len(testIndices))

	for i, idx := range trainIndices {
		XTrain[i] = make([]decimal.Decimal, len(X[idx]))
		copy(XTrain[i], X[idx])
		yTrain[i] = y[idx]
	}

	for i, idx := range testIndices {
		XTest[i] = make([]decimal.Decimal, len(X[idx]))
		copy(XTest[i], X[idx])
		yTest[i] = y[idx]
	}

	return XTrain, XTest, yTrain, yTest, nil
}

func DefaultTrainTestSplitter() *TrainTestSplitter {
	return NewTrainTestSplitter(0.2, time.Now().UnixNano(), true)
}

type KFoldSplitter struct {
	nFolds     int
	shuffle    bool
	randomSeed int64
}

func NewKFoldSplitter(nFolds int, shuffle bool, randomSeed int64) *KFoldSplitter {
	return &KFoldSplitter{
		nFolds:     nFolds,
		shuffle:    shuffle,
		randomSeed: randomSeed,
	}
}

func (kfs *KFoldSplitter) Split(X [][]decimal.Decimal, y []int) ([][][]decimal.Decimal, [][][]decimal.Decimal, [][]int, [][]int, error) {
	if len(X) != len(y) {
		return nil, nil, nil, nil, fmt.Errorf("x and y must have the same length")
	}

	if len(X) == 0 {
		return nil, nil, nil, nil, fmt.Errorf("cannot split empty dataset")
	}

	if kfs.nFolds <= 1 || kfs.nFolds > len(X) {
		return nil, nil, nil, nil, fmt.Errorf("number of folds must be between 2 and %d", len(X))
	}

	n := len(X)
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	if kfs.shuffle {
		rng := rand.New(rand.NewSource(kfs.randomSeed))
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	var XTrainFolds, XTestFolds [][][]decimal.Decimal
	var yTrainFolds, yTestFolds [][]int

	foldSize := n / kfs.nFolds

	for fold := 0; fold < kfs.nFolds; fold++ {
		testStart := fold * foldSize
		testEnd := testStart + foldSize
		if fold == kfs.nFolds-1 {
			testEnd = n
		}

		testIndices := indices[testStart:testEnd]
		var trainIndices []int
		trainIndices = append(trainIndices, indices[:testStart]...)
		trainIndices = append(trainIndices, indices[testEnd:]...)

		XTrain := make([][]decimal.Decimal, len(trainIndices))
		XTest := make([][]decimal.Decimal, len(testIndices))
		yTrain := make([]int, len(trainIndices))
		yTest := make([]int, len(testIndices))

		for i, idx := range trainIndices {
			XTrain[i] = make([]decimal.Decimal, len(X[idx]))
			copy(XTrain[i], X[idx])
			yTrain[i] = y[idx]
		}

		for i, idx := range testIndices {
			XTest[i] = make([]decimal.Decimal, len(X[idx]))
			copy(XTest[i], X[idx])
			yTest[i] = y[idx]
		}

		XTrainFolds = append(XTrainFolds, XTrain)
		XTestFolds = append(XTestFolds, XTest)
		yTrainFolds = append(yTrainFolds, yTrain)
		yTestFolds = append(yTestFolds, yTest)
	}

	return XTrainFolds, XTestFolds, yTrainFolds, yTestFolds, nil
}