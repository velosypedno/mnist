package dataset

import (
	"encoding/binary"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

func loadImagesAndLabels(imagesPath, labelsPath string) ([][]byte, []uint8, error) {
	lFile, err := os.Open(labelsPath)
	if err != nil {
		return nil, nil, err
	}
	defer lFile.Close()

	var lMagic, lCount uint32
	binary.Read(lFile, binary.BigEndian, &lMagic)
	binary.Read(lFile, binary.BigEndian, &lCount)
	if lMagic != 2049 {
		return nil, nil, fmt.Errorf("wrong magic number for labels: %d", lMagic)
	}

	labels := make([]uint8, lCount)
	if _, err := lFile.Read(labels); err != nil {
		return nil, nil, err
	}

	iFile, err := os.Open(imagesPath)
	if err != nil {
		return nil, nil, err
	}
	defer iFile.Close()

	var iMagic, iCount, rows, cols uint32
	binary.Read(iFile, binary.BigEndian, &iMagic)
	binary.Read(iFile, binary.BigEndian, &iCount)
	binary.Read(iFile, binary.BigEndian, &rows)
	binary.Read(iFile, binary.BigEndian, &cols)

	if iMagic != 2051 {
		return nil, nil, fmt.Errorf("wrong magic number for images: %d", iMagic)
	}

	if iCount != lCount {
		return nil, nil, fmt.Errorf("labels count %d != images count %d", lCount, iCount)
	}

	images := make([][]byte, iCount)
	pixelCount := int(rows * cols)
	for i := 0; i < int(iCount); i++ {
		img := make([]byte, pixelCount)
		iFile.Read(img)
		images[i] = img
	}

	return images, labels, nil
}

func Load(imagesPath, labelsPath string) (*mat.Dense, *mat.Dense, error) {
	images, labels, err := loadImagesAndLabels(imagesPath, labelsPath)
	if err != nil {
		return nil, nil, err
	}

	n := len(labels)

	const inputDim = 784
	const outputDim = 10

	x := mat.NewDense(n, inputDim, nil)
	y := mat.NewDense(n, outputDim, nil)

	for i := 0; i < n; i++ {
		xRow := x.RawRowView(i)
		yRow := y.RawRowView(i)

		for j := 0; j < inputDim; j++ {
			xRow[j] = float64(images[i][j]) / 255.0
		}

		label := labels[i]
		if label < outputDim {
			yRow[label] = 1.0
		}
	}

	return x, y, nil

}
