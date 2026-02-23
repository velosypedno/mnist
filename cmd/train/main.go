package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"

	"github.com/velosypedno/mnist/dataset"
	"github.com/velosypedno/nns/layer"
	"github.com/velosypedno/nns/logger"
	"github.com/velosypedno/nns/loss"
	"github.com/velosypedno/nns/network"
)

func init() {
	rand.Seed(42)
}

func main() {
	imgPtr := flag.String("images", "", "Path to MNIST images file")
	lblPtr := flag.String("labels", "", "Path to MNIST labels file")
	outPtr := flag.String("output", "./models/mnist.gob", "Path to save the trained model")

	flag.Parse()

	if *imgPtr == "" || *lblPtr == "" {
		fmt.Println("Error: images and labels paths are required")
		flag.Usage()
		os.Exit(1)
	}

	layers := []network.Layer{
		layer.NewDense(784, 392),
		layer.NewTanh(),
		layer.NewDense(392, 196),
		layer.NewTanh(),
		layer.NewDense(196, 10),
	}
	lr := 0.05

	lgr := logger.NewPrettyLogger()
	defer lgr.Sync()

	n := network.New(
		layers,
		lr,
		loss.NewSoftMaxCrossEntropyFunc(),
		network.WithLogger(lgr),
		network.WithLogInterval(1),
		network.WithBatchSize(32),
		network.WithEpochs(30),
	)

	X, Y, err := dataset.Load(*imgPtr, *lblPtr)
	if err != nil {
		panic(err)
	}

	trainSize, _ := X.Dims()
	fmt.Printf("Training for MNIST with %d samples...\n", trainSize)

	n.Fit(X, Y)

	err = n.SaveToFile(*outPtr)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Model saved to %s\n", *outPtr)
}
