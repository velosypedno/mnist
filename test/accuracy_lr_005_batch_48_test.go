package test

import (
	"fmt"
	"testing"

	"github.com/velosypedno/mnist/dataset"
	"github.com/velosypedno/nns/network"
)

func TestAccuracyLR005Batch48(t *testing.T) {
	n, err := network.LoadFromFile("../models/lr005batch48.gob")
	if err != nil {
		panic(err)
	}

	testX, testY, err := dataset.Load(
		"../data/t10k-images.idx3-ubyte",
		"../data/t10k-labels.idx1-ubyte",
	)
	if err != nil {
		panic(err)
	}

	fmt.Println("\n--- Running MNIST Evaluation ---")
	predictions := n.Predict(testX)

	total, _ := predictions.Dims()
	correct := 0
	for i := 0; i < total; i++ {
		if argmax(predictions.RawRowView(i)) == argmax(testY.RawRowView(i)) {
			correct++
		}
	}

	accuracy := float64(correct) / float64(total) * 100
	fmt.Printf("Total Test Samples: %d | Accuracy: %.2f%%\n", total, accuracy)
}

func argmax(slice []float64) int {
	maxIdx := 0
	maxVal := slice[0]
	for i, v := range slice {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
