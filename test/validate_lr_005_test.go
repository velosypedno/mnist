package test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/velosypedno/mnist/dataset"
	"github.com/velosypedno/nns/network"
)

func TestValidateLR005(t *testing.T) {
	n, err := network.LoadFromFile("../models/lr005.gob")
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

	fmt.Println("\n--- Visualizing First 5 Samples ---")
	for i := 0; i < 10; i++ {
		rowX := testX.RawRowView(i)
		rowY := testY.RawRowView(i)
		rowP := predictions.RawRowView(i)

		target := argmax(rowY)
		pred := argmax(rowP)
		conf := rowP[pred] * 100

		fmt.Printf("\nSample #%d | Target: %d | Pred: %d | Confidence: %.2f%%\n", i, target, pred, conf)
		drawDigit(rowX)
		fmt.Println(strings.Repeat("-", 30))
	}

}

func drawDigit(pixels []float64) {
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			p := pixels[i*28+j]
			if p > 0.8 {
				fmt.Print("##")
			} else if p > 0.4 {
				fmt.Print("++")
			} else if p > 0.1 {
				fmt.Print("..")
			} else {
				fmt.Print("  ")
			}
		}
		fmt.Println()
	}
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
