package jobs

import (
    "fmt"
    "sync"
    "time"
)

type JobStatus string

const (
    JobPending   JobStatus = "pending"
    JobRunning   JobStatus = "running"
    JobCompleted JobStatus = "completed"
    JobFailed    JobStatus = "failed"
    JobCancelled JobStatus = "cancelled"
)

type Job struct {
    ID          string
    Type        string
    Status      JobStatus
    Progress    float64
    StartTime   time.Time
    EndTime     *time.Time
    Error       error
    Result      any
    Description string
    Logs        []string
    cancelFunc  func()
    mu          sync.RWMutex
}

type Manager struct {
    jobs map[string]*Job
    mu   sync.RWMutex
}

func NewManager() *Manager {
    return &Manager{
        jobs: make(map[string]*Job),
    }
}

func (m *Manager) CreateJob(jobType, description string) *Job {
    m.mu.Lock()
    defer m.mu.Unlock()

    jobID := fmt.Sprintf("job_%s_%d", jobType, time.Now().UnixNano())
    job := &Job{
        ID:          jobID,
        Type:        jobType,
        Status:      JobPending,
        StartTime:   time.Now(),
        Description: description,
        Logs:        []string{},
    }

    m.jobs[jobID] = job
    return job
}

func (m *Manager) GetJob(jobID string) (*Job, bool) {
    m.mu.RLock()
    defer m.mu.RUnlock()

    job, exists := m.jobs[jobID]
    return job, exists
}

func (m *Manager) ListJobs() []*Job {
    m.mu.RLock()
    defer m.mu.RUnlock()

    jobs := make([]*Job, 0, len(m.jobs))
    for _, job := range m.jobs {
        jobs = append(jobs, job)
    }
    return jobs
}

func (m *Manager) CancelJob(jobID string) error {
    job, exists := m.GetJob(jobID)
    if !exists {
        return fmt.Errorf("job %s not found", jobID)
    }

    job.mu.Lock()
    defer job.mu.Unlock()

    if job.Status != JobRunning {
        return fmt.Errorf("job %s is not running", jobID)
    }

    if job.cancelFunc != nil {
        job.cancelFunc()
        job.Status = JobCancelled
        now := time.Now()
        job.EndTime = &now
    }

    return nil
}

func (j *Job) SetStatus(status JobStatus) {
    j.mu.Lock()
    defer j.mu.Unlock()
    j.Status = status
    if status == JobCompleted || status == JobFailed || status == JobCancelled {
        now := time.Now()
        j.EndTime = &now
    }
}

func (j *Job) SetProgress(progress float64) {
    j.mu.Lock()
    defer j.mu.Unlock()
    j.Progress = progress
}

func (j *Job) AddLog(message string) {
    j.mu.Lock()
    defer j.mu.Unlock()
    timestamp := time.Now().Format("15:04:05")
    j.Logs = append(j.Logs, fmt.Sprintf("[%s] %s", timestamp, message))
}

func (j *Job) SetError(err error) {
    j.mu.Lock()
    defer j.mu.Unlock()
    j.Error = err
    j.Status = JobFailed
    now := time.Now()
    j.EndTime = &now
}

func (j *Job) SetResult(result any) {
    j.mu.Lock()
    defer j.mu.Unlock()
    j.Result = result
}

func (j *Job) SetCancelFunc(cancelFunc func()) {
    j.mu.Lock()
    defer j.mu.Unlock()
    j.cancelFunc = cancelFunc
}

func (j *Job) GetStatus() JobStatus {
    j.mu.RLock()
    defer j.mu.RUnlock()
    return j.Status
}

func (j *Job) GetProgress() float64 {
    j.mu.RLock()
    defer j.mu.RUnlock()
    return j.Progress
}

func (j *Job) GetLogs() []string {
    j.mu.RLock()
    defer j.mu.RUnlock()
    logs := make([]string, len(j.Logs))
    copy(logs, j.Logs)
    return logs
}