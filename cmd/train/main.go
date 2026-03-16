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
	"github.com/velosypedno/nns/network/cnn"
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

	classificationLayers := []cnn.MLPLayer{
		layer.NewDense(400, 200),
		layer.NewTanh(),
		layer.NewDense(200, 100),
		layer.NewTanh(),
		layer.NewDense(100, 10),
	}

	convolutionLayers := []cnn.CNNLayer{
		layer.NewConv(3, 8, 1, 28, 28),
		layer.NewReLU(),
		layer.NewMaxPool(2, 2, 8, 26, 26),
		layer.NewReLU(),
		layer.NewConv(3, 16, 8, 13, 13),
		layer.NewReLU(),
		layer.NewMaxPool(2, 2, 16, 11, 11),
	}

	lgr := logger.NewPrettyLogger()
	defer lgr.Sync()

	n := cnn.New(
		convolutionLayers,
		classificationLayers,
		cnn.WithLoss(loss.NewSoftMaxCrossEntropyFunc()),
		cnn.WithLogger(lgr),
		cnn.WithLogInterval(1),
		cnn.WithBatchSize(32),
		cnn.WithEpochs(16),
		cnn.WithLearningRate(0.05),
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
