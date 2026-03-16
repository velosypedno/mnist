package main

import (
	"fmt"
	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
	"github.com/velosypedno/nns/network/cnn"
	"gonum.org/v1/gonum/mat"
)

const gridSize = 28

type drawableGrid struct {
	widget.BaseWidget
	rects     [gridSize][gridSize]*canvas.Rectangle
	pixels    [gridSize][gridSize]float32
	onChanged func([gridSize][gridSize]float32)
	gridObj   *fyne.Container
}

func newDrawableGrid(onChanged func([gridSize][gridSize]float32)) *drawableGrid {
	g := &drawableGrid{onChanged: onChanged}
	g.ExtendBaseWidget(g)

	g.gridObj = container.New(layout.NewGridLayout(gridSize))
	for y := 0; y < gridSize; y++ {
		for x := 0; x < gridSize; x++ {
			rect := canvas.NewRectangle(color.White)
			rect.StrokeColor = color.NRGBA{235, 235, 235, 255}
			rect.StrokeWidth = 0.5
			g.rects[y][x] = rect
			g.gridObj.Add(rect)
		}
	}
	return g
}

func (g *drawableGrid) CreateRenderer() fyne.WidgetRenderer {
	return widget.NewSimpleRenderer(g.gridObj)
}
func (g *drawableGrid) MinSize() fyne.Size {
	return fyne.NewSize(600, 600)
}
func (g *drawableGrid) Dragged(e *fyne.DragEvent) {
	size := g.Size()
	cellWidth := size.Width / float32(gridSize)
	cellHeight := size.Height / float32(gridSize)

	centerX := int(e.Position.X / cellWidth)
	centerY := int(e.Position.Y / cellHeight)

	changed := false

	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			x, y := centerX+dx, centerY+dy

			if x >= 0 && x < gridSize && y >= 0 && y < gridSize {
				distSq := dx*dx + dy*dy
				var intensity float32

				switch distSq {
				case 0:
					intensity = 1.0
				case 1:
					intensity = 0.4
				case 2:
					intensity = 0.1
				default:
					continue
				}

				if intensity > g.pixels[y][x] {
					g.pixels[y][x] = intensity
					c := uint8(255 * (1 - intensity))
					g.rects[y][x].FillColor = color.NRGBA{c, c, c, 255}
					g.rects[y][x].Refresh()
					changed = true
				}
			}
		}
	}

	if changed {
		g.onChanged(g.pixels)
	}
}

func (g *drawableGrid) DragEnd() {}

func (g *drawableGrid) Clear() {
	for y := 0; y < gridSize; y++ {
		for x := 0; x < gridSize; x++ {
			g.pixels[y][x] = 0
			g.rects[y][x].FillColor = color.White
			g.rects[y][x].Refresh()
		}
	}
	g.onChanged(g.pixels)
}

func predict(n *cnn.CNN, data [gridSize][gridSize]float32) []float32 {
	flatData := make([]float64, gridSize*gridSize)
	for y := 0; y < gridSize; y++ {
		for x := 0; x < gridSize; x++ {
			flatData[y*gridSize+x] = float64(data[y][x])
		}
	}

	inputs := mat.NewDense(1, gridSize*gridSize, flatData)

	output := n.Predict(inputs)

	res := make([]float32, 10)
	for i := 0; i < 10; i++ {
		res[i] = float32(output.At(0, i))
	}

	return res
}

func main() {
	n, err := cnn.LoadFromFile("./models/mnist.gob")
	if err != nil {
		panic(err)
	}

	myApp := app.New()
	myApp.Settings().SetTheme(theme.LightTheme())
	myWindow := myApp.NewWindow("Digit Recognizer")

	labels := make([]*widget.Label, 10)
	progressBars := make([]*widget.ProgressBar, 10)
	for i := 0; i < 10; i++ {
		labels[i] = widget.NewLabel(fmt.Sprintf("%d: 0.00", i))
		progressBars[i] = widget.NewProgressBar()
	}

	updatePredictions := func(data [gridSize][gridSize]float32) {
		probs := predict(n, data)
		for i := 0; i < 10; i++ {
			labels[i].SetText(fmt.Sprintf("%d: %.2f", i, probs[i]))
			progressBars[i].SetValue(float64(probs[i]))
		}
	}

	drawingArea := newDrawableGrid(updatePredictions)

	sidebar := container.NewVBox()
	sidebar.Add(widget.NewLabelWithStyle("Probabilities", fyne.TextAlignCenter, fyne.TextStyle{Bold: true}))

	for i := 0; i < 10; i++ {
		barContainer := container.NewStack(progressBars[i])
		row := container.NewBorder(nil, nil, labels[i], nil, barContainer)
		sidebar.Add(row)
	}

	sidebarWrapper := container.NewStack(sidebar)

	clearBtn := widget.NewButtonWithIcon("Clear Canvas", theme.DeleteIcon(), func() {
		drawingArea.Clear()
	})

	rightPanel := container.NewBorder(nil, clearBtn, nil, nil, sidebarWrapper)

	mainLayout := container.NewHBox(
		drawingArea,
		widget.NewSeparator(),
		container.NewPadded(rightPanel),
	)

	centeredContent := container.NewCenter(mainLayout)
	myWindow.SetContent(centeredContent)
	myWindow.Resize(fyne.NewSize(1000, 750))
	myWindow.ShowAndRun()
}
