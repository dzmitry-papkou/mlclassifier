package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"

	"github.com/shopspring/decimal"
)

type StreamingCSVReader struct {
	file   *os.File
	reader *csv.Reader
	header []string
}

func NewStreamingCSVReader(filename string) (*StreamingCSVReader, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(file)

	header, err := reader.Read()
	if err != nil {
		file.Close()
		return nil, err
	}

	return &StreamingCSVReader{
		file:   file,
		reader: reader,
		header: header,
	}, nil
}

func (r *StreamingCSVReader) GetHeaders() []string {
	return r.header
}

func (r *StreamingCSVReader) ReadBatch(batchSize int) ([][]decimal.Decimal, []int, error) {
	var X [][]decimal.Decimal
	var y []int

	for i := 0; i < batchSize; i++ {
		record, err := r.reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}

		features := make([]decimal.Decimal, len(record)-1)
		for j := 0; j < len(record)-1; j++ {
			val, err := decimal.NewFromString(record[j])
			if err != nil {
				return nil, nil, fmt.Errorf("invalid numeric value at column %d: %s", j, record[j])
			}
			features[j] = val
		}

		label := record[len(record)-1]
		labelInt := hashStringToInt(label)

		X = append(X, features)
		y = append(y, labelInt)
	}

	return X, y, nil
}

func (r *StreamingCSVReader) Close() error {
	return r.file.Close()
}

func hashStringToInt(s string) int {
	hash := 0
	for _, char := range s {
		hash = hash*31 + int(char)
	}
	if hash < 0 {
		hash = -hash
	}
	return hash % 1000
}