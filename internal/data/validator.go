package data

import (
	"fmt"

	"github.com/shopspring/decimal"
)

type DataValidator struct{}

func NewDataValidator() *DataValidator {
	return &DataValidator{}
}

func (dv *DataValidator) ValidateDataset(X [][]decimal.Decimal, y []int) error {
	if len(X) == 0 {
		return fmt.Errorf("dataset is empty")
	}

	if len(X) != len(y) {
		return fmt.Errorf("feature matrix and labels have different lengths: %d vs %d", len(X), len(y))
	}

	nFeatures := len(X[0])
	if nFeatures == 0 {
		return fmt.Errorf("features cannot be empty")
	}

	for i, sample := range X {
		if len(sample) != nFeatures {
			return fmt.Errorf("inconsistent feature count at sample %d: expected %d, got %d", i, nFeatures, len(sample))
		}
	}

	for i, sample := range X {
		for j, value := range sample {
			if value.IsZero() && value.String() == "NaN" {
				return fmt.Errorf("missing value (NaN) at sample %d, feature %d", i, j)
			}
		}
	}

	return nil
}

func (dv *DataValidator) ValidateLabels(y []int) error {
	if len(y) == 0 {
		return fmt.Errorf("labels are empty")
	}

	classCount := make(map[int]int)
	for _, label := range y {
		classCount[label]++
	}

	if len(classCount) < 2 {
		return fmt.Errorf("dataset must have at least 2 classes, found %d", len(classCount))
	}

	return nil
}

func (dv *DataValidator) ValidateTrainTestSplit(XTrain, XTest [][]decimal.Decimal, yTrain, yTest []int) error {
	if err := dv.ValidateDataset(XTrain, yTrain); err != nil {
		return fmt.Errorf("training set validation failed: %v", err)
	}

	if err := dv.ValidateDataset(XTest, yTest); err != nil {
		return fmt.Errorf("test set validation failed: %v", err)
	}

	if len(XTrain[0]) != len(XTest[0]) {
		return fmt.Errorf("train and test sets have different feature counts: %d vs %d", len(XTrain[0]), len(XTest[0]))
	}

	return nil
}

func (dv *DataValidator) GetDatasetStats(X [][]decimal.Decimal, y []int) map[string]any {
	if len(X) == 0 {
		return map[string]any{}
	}

	stats := make(map[string]any)
	stats["samples"] = len(X)
	stats["features"] = len(X[0])

	classCount := make(map[int]int)
	for _, label := range y {
		classCount[label]++
	}
	stats["classes"] = len(classCount)
	stats["class_distribution"] = classCount

	nFeatures := len(X[0])
	featureStats := make([]map[string]decimal.Decimal, nFeatures)

	for j := 0; j < nFeatures; j++ {
		values := make([]decimal.Decimal, len(X))
		for i := 0; i < len(X); i++ {
			values[i] = X[i][j]
		}

		featureStats[j] = map[string]decimal.Decimal{
			"min":  findMin(values),
			"max":  findMax(values),
			"mean": calculateMean(values),
		}
	}
	stats["feature_stats"] = featureStats

	return stats
}

func findMin(values []decimal.Decimal) decimal.Decimal {
	if len(values) == 0 {
		return decimal.Zero
	}
	min := values[0]
	for _, v := range values[1:] {
		if v.LessThan(min) {
			min = v
		}
	}
	return min
}

func findMax(values []decimal.Decimal) decimal.Decimal {
	if len(values) == 0 {
		return decimal.Zero
	}
	max := values[0]
	for _, v := range values[1:] {
		if v.GreaterThan(max) {
			max = v
		}
	}
	return max
}

func calculateMean(values []decimal.Decimal) decimal.Decimal {
	if len(values) == 0 {
		return decimal.Zero
	}
	sum := decimal.Zero
	for _, v := range values {
		sum = sum.Add(v)
	}
	return sum.Div(decimal.NewFromInt(int64(len(values))))
}