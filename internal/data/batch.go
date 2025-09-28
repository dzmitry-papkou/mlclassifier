package data

import (
	"github.com/shopspring/decimal"
)

type BatchProcessor struct {
	batchSize int
}

func NewBatchProcessor(batchSize int) *BatchProcessor {
	return &BatchProcessor{batchSize: batchSize}
}

func (bp *BatchProcessor) ProcessBatches(X [][]decimal.Decimal, y []int, processFn func([][]decimal.Decimal, []int) error) error {
	totalSamples := len(X)

	for start := 0; start < totalSamples; start += bp.batchSize {
		end := start + bp.batchSize
		if end > totalSamples {
			end = totalSamples
		}

		batchX := X[start:end]
		batchY := y[start:end]

		if err := processFn(batchX, batchY); err != nil {
			return err
		}
	}

	return nil
}

func (bp *BatchProcessor) SetBatchSize(size int) {
	bp.batchSize = size
}

func (bp *BatchProcessor) GetBatchSize() int {
	return bp.batchSize
}