package main

import (
    "flag"
    "mlclassifier/internal/commander"
)

func main() {
    interactive := flag.Bool("i", true, "Interactive mode")
    flag.Parse()
    
    if *interactive {
        cmd := commander.NewCommander()
        cmd.Start()
    }
}