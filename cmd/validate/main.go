package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/velosypedno/mnist/dataset"
	"github.com/velosypedno/nns/network"
)

func main() {
	imgPtr := flag.String("images", "", "Path to test images file")
	lblPtr := flag.String("labels", "", "Path to test labels file")
	modelPtr := flag.String("model", "./models/mnist.gob", "Path to the trained model file")
	visualizePtr := flag.Int("n", 10, "Number of samples to visualize")

	flag.Parse()

	if *imgPtr == "" || *lblPtr == "" {
		fmt.Println("Error: images and labels paths are required")
		flag.Usage()
		os.Exit(1)
	}

	n, err := network.LoadFromFile(*modelPtr)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		os.Exit(1)
	}

	testX, testY, err := dataset.Load(*imgPtr, *lblPtr)
	if err != nil {
		fmt.Printf("Error loading dataset: %v\n", err)
		os.Exit(1)
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

	if *visualizePtr > 0 {
		fmt.Printf("\n--- Visualizing First %d Samples ---\n", *visualizePtr)
		for i := 0; i < *visualizePtr; i++ {
			rowX := testX.RawRowView(i)
			rowY := testY.RawRowView(i)
			rowP := predictions.RawRowView(i)

			target := argmax(rowY)
			pred := argmax(rowP)
			conf := rowP[pred] * 100

			status := "CORRECT"
			if target != pred {
				status := "WRONG"
				_ = status
			}

			fmt.Printf("\nSample #%d [%s] | Target: %d | Pred: %d | Confidence: %.2f%%\n",
				i, status, target, pred, conf)

			drawDigit(rowX)
			fmt.Println(strings.Repeat("-", 56))
		}
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
