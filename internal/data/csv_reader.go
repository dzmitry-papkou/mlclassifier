package data

import (
    "encoding/csv"
    "fmt"
    "io"
    "os"
    "strings"
    
    "github.com/shopspring/decimal"
    "mlclassifier/internal/preprocessing"
)

type DataBatch struct {
    X      [][]decimal.Decimal
    Labels []string
    Size   int
}

type StreamingReader struct {
    file       *os.File
    reader     *csv.Reader
    headers    []string
    labelCol   int
    batchSize  int
    encoder    *preprocessing.LabelEncoder
}

func NewStreamingReader(filename string, labelCol int, batchSize int) (*StreamingReader, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open file: %w", err)
    }
    
    reader := csv.NewReader(file)
    
    headers, err := reader.Read()
    if err != nil {
        file.Close()
        return nil, fmt.Errorf("failed to read headers: %w", err)
    }
    
    if labelCol < 0 || labelCol >= len(headers) {
        labelCol = len(headers) - 1
    }
    
    return &StreamingReader{
        file:      file,
        reader:    reader,
        headers:   headers,
        labelCol:  labelCol,
        batchSize: batchSize,
        encoder:   preprocessing.NewLabelEncoder(),
    }, nil
}

func (sr *StreamingReader) ReadBatch() (*DataBatch, error) {
    batch := &DataBatch{
        X:      make([][]decimal.Decimal, 0, sr.batchSize),
        Labels: make([]string, 0, sr.batchSize),
    }
    
    for i := 0; i < sr.batchSize; i++ {
        record, err := sr.reader.Read()
        if err == io.EOF {
            if len(batch.X) == 0 {
                return nil, io.EOF
            }
            break
        }
        if err != nil {
            return nil, fmt.Errorf("error reading record: %w", err)
        }
        
        hasEmpty := false
        for _, val := range record {
            if strings.TrimSpace(val) == "" {
                hasEmpty = true
                break
            }
        }
        if hasEmpty {
            i--
            continue
        }
        
        features := make([]decimal.Decimal, 0, len(record)-1)
        label := ""
        
        for j, val := range record {
            if j == sr.labelCol {
                label = val
            } else {
                decVal, err := decimal.NewFromString(val)
                if err != nil {
                    decVal = decimal.Zero
                }
                features = append(features, decVal)
            }
        }
        
        batch.X = append(batch.X, features)
        batch.Labels = append(batch.Labels, label)
    }
    
    batch.Size = len(batch.X)
    return batch, nil
}

func (sr *StreamingReader) GetHeaders() []string {
    return sr.headers
}

func (sr *StreamingReader) GetEncoder() *preprocessing.LabelEncoder {
    return sr.encoder
}

func (sr *StreamingReader) Close() error {
    return sr.file.Close()
}

func ProcessLargeFile(filename string, processor func(*DataBatch) error) error {
    reader, err := NewStreamingReader(filename, -1, 1000)
    if err != nil {
        return err
    }
    defer reader.Close()

    batchNum := 0
    for {
        batch, err := reader.ReadBatch()
        if err == io.EOF {
            break
        }
        if err != nil {
            return fmt.Errorf("error reading batch %d: %w", batchNum, err)
        }

        if err := processor(batch); err != nil {
            return fmt.Errorf("error processing batch %d: %w", batchNum, err)
        }

        batchNum++
    }

    return nil
}

type CSVReader struct {
    filename string
}

func NewCSVReader(filename string) (*CSVReader, error) {
    return &CSVReader{filename: filename}, nil
}

func (cr *CSVReader) LoadData() ([][]decimal.Decimal, []int, []string, error) {
    file, err := os.Open(cr.filename)
    if err != nil {
        return nil, nil, nil, err
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        return nil, nil, nil, err
    }

    if len(records) < 2 {
        return nil, nil, nil, fmt.Errorf("insufficient data in file")
    }

    headers := records[0][:len(records[0])-1]
    data := records[1:]

    X := make([][]decimal.Decimal, len(data))
    labels := make([]string, len(data))

    for i, record := range data {
        X[i] = make([]decimal.Decimal, len(record)-1)
        for j := 0; j < len(record)-1; j++ {
            val, err := decimal.NewFromString(record[j])
            if err != nil {
                val = decimal.Zero
            }
            X[i][j] = val
        }
        labels[i] = record[len(record)-1]
    }

    encoder := preprocessing.NewLabelEncoder()
    y, err := encoder.FitTransform(labels)
    if err != nil {
        return nil, nil, nil, err
    }

    return X, y, headers, nil
}