package test

import (
	"fmt"
	"testing"

	"github.com/velosypedno/mnist/dataset"
	"github.com/velosypedno/nns/layer"
	"github.com/velosypedno/nns/logger"
	"github.com/velosypedno/nns/loss"
	"github.com/velosypedno/nns/network"
)

func TestTrainLearningRate002(t *testing.T) {
	layers := []network.Layer{
		layer.NewDense(784, 392),
		layer.NewTanh(),
		layer.NewDense(392, 196),
		layer.NewTanh(),
		layer.NewDense(196, 10),
	}

	lr := 0.02
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

	X, Y, err := dataset.Load(
		"../data/train-images-idx3-ubyte/train-images-idx3-ubyte",
		"../data/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
	)
	if err != nil {
		t.Fatal(err)
	}

	trainSize, _ := X.Dims()
	fmt.Printf("Training for MNIST with %d samples...\n", trainSize)
	n.Fit(X, Y)

	err = n.SaveToFile("../models/lr002.gob")
	if err != nil {
		t.Fatal(err)
	}
}
