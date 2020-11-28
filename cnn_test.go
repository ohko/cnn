package neatgo

import (
	"cnn/mnist"
	"fmt"
	"log"
	"testing"
)

// go test cnn -run TestCNN -v -count=1
func TestCNN(t *testing.T) {
	input := normalization([][]float64{
		{1, 2, 3, 2, 1},
		{4, 5, 6, 5, 4},
		{7, 8, 9, 8, 7},
		{4, 5, 6, 5, 4},
		{1, 2, 3, 2, 1},
	})
	wants := []float64{1, 0}

	conv := NewConvolution(3, 3, 2, 1)
	pool := NewMaxPool(2, 2)
	learnRate := 0.3

	iw, ih := len(input[0]), len(input)
	sw, sh := SoftmaxHelper(iw, ih, conv, pool)
	fmt.Printf("input size: %d*%d\n", iw, ih)
	fmt.Printf("conv size: %d*%d*%d step:%d\n", conv.w, conv.h, conv.depth, conv.step)
	fmt.Printf("pool size: %d*%d\n", pool.w, pool.h)
	fmt.Printf("softmax size: %d*%d\n", sw, sh)
	fmt.Printf("learn rate: %.4f\n", learnRate)
	softmax := NewSoftmax(sw, sh, conv.depth, len(wants))

	count := 10000
	debug := false
	if count == 1 {
		debug = true
		fmt.Print("\033c")
	}
	for c := 0; c < count; c++ {
		out := conv.Forward(input)
		print3(debug, "conv", out)

		out2 := pool.Forward(out)
		print3(debug, "pool", out2)

		outputs := softmax.Forward(out2)
		print1(debug, "outputs:", outputs)

		if c%1000 == 0 {
			fmt.Printf("LOSS: %d %.8f, %.8f\n", c, wants, outputs)
		}

		if debug {
			fmt.Println("\n===== BACKPROP =====")
		}

		gradient := softmax.Backprop(outputs, wants, learnRate)
		print3(debug, "softmax.backprop", gradient)

		gradient = pool.Backprop(gradient)
		print3(debug, "pool.backprop", gradient)

		gradient = conv.Backprop(gradient, pool.maxIndex, learnRate)
		print3(debug, "conv.filters", gradient)
	}
}

// go test cnn -run TestMnist -v -count=1
func TestMnist(t *testing.T) {
	dataTrain, err := mnist.ReadTrainSet("./mnist/MNIST_data")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("MNISST train: N:%v | W:%v | H:%v", dataTrain.N, dataTrain.W, dataTrain.H)

	conv := NewConvolution(3, 3, 1, 11)
	pool := NewMaxPool(2, 2)
	softmax := NewSoftmax(13, 13, 1, 10)

	debug := false
	count := 2
	for ccc := 0; ccc < 5; ccc++ {
		fmt.Println("\nccc:", ccc)
		right := 0
		for c, v := range dataTrain.Data {
			bits := normalization(Uint8ToFloat64(v.Image))
			wants := make([]float64, 10)
			wants[v.Digit] = 1

			out := conv.Forward(bits)
			if count == 1 {
				print3(debug, "conv", out)
			}

			out2 := pool.Forward(out)
			if count == 1 {
				print3(debug, "pool", out2)
			}

			if count == 1 {
				print2(debug, "softmax.weights:", softmax.weights)
			}
			outputs := softmax.Forward(out2)
			if count == 1 {
				print1(debug, "outputs:", outputs)
			}
			maxI, maxV := GetMax(outputs)
			if v.Digit == maxI && maxV > 0.9 {
				right++
			}
			if c%100 == 0 {
				fmt.Printf("\rLOSS:[%.3f%%] %d %d %.2f", float64(right)/float64(c)*100, c, v.Digit, outputs)
			}

			if count == 1 {
				fmt.Println("\n===== BACKPROP =====")
			}

			learnRate := 0.6
			gradient := softmax.Backprop(outputs, wants, learnRate)
			if count == 1 {
				print3(debug, "softmax.backprop", gradient)
			}
			gradient = pool.Backprop(gradient)
			if count == 1 {
				print3(debug, "pool.backprop", gradient)
			}
			// gradient = conv.Backprop(rotate180(gradient), learnRate)
			gradient = conv.Backprop(gradient, pool.maxIndex, learnRate)
			if count == 1 {
				print3(debug, "conv.filters", gradient)
			}
			// time.Sleep(time.Second * 10)
		}
	}

	dataCheck, err := mnist.ReadTestSet("./mnist/MNIST_data")
	if err != nil {
		log.Fatal(err)
	}
	{
		right := 0
		for c2, v := range dataCheck.Data {
			bits := normalization(Uint8ToFloat64(v.Image))
			wants := make([]float64, 10)
			wants[v.Digit] = 1

			out := conv.Forward(bits)
			if count == 1 {
				print3(debug, "conv", out)
			}

			out2 := pool.Forward(out)
			if count == 1 {
				print3(debug, "pool", out2)
			}

			if count == 1 {
				print2(debug, "softmax.weights:", softmax.weights)
			}
			outputs := softmax.Forward(out2)
			if count == 1 {
				print1(debug, "outputs:", outputs)
			}
			maxI, _ := GetMax(outputs)
			if v.Digit == maxI {
				right++
			}
			fmt.Printf("\rLOSS:[%.3f%%]", float64(right)/float64(c2)*100)
		}
	}
}

// go test cnn -run Test1 -v -count=1
func Test1(t *testing.T) {
	cnn := func(img, filter [][]float64) [][]float64 {
		out := [][]float64{}
		for y := 0; y <= len(img)-len(filter); y++ {
			row := []float64{}
			for x := 0; x <= len(img[0])-len(filter[0]); x++ {
				sum := 0.0
				// htm := []string{}
				for b := 0; b < len(filter); b++ {
					for a := 0; a < len(filter[0]); a++ {
						sum += img[y+b][x+a] * filter[b][a]
						// htm = append(htm, fmt.Sprintf("%.4f*%.4f=%.4f", img[y+b][x+a], filter[b][a], img[y+b][x+a]*filter[b][a]))
					}
				}
				row = append(row, sum)
				// fmt.Println(strings.Join(htm, " + "), sum)
			}
			out = append(out, row)
		}

		return out
	}

	d1 := [][]float64{
		{0.51, 0.9, 0.88, 0.84, 0.05},
		{0.4, 0.62, 0.22, 0.59, 0.1},
		{0.11, 0.2, 0.74, 0.33, 0.14},
		{0.47, 0.01, 0.85, 0.7, 0.09},
		{0.76, 0.19, 0.72, 0.17, 0.57},
	}
	d2 := [][][]float64{
		{
			{0, 0, 0.0686, 0},
			{0, 0.0364, 0, 0},
			{0, 0.0467, 0, 0},
			{0, 0, 0, -0.0681},
		},
	}

	d2r := rotate180(d2)
	fmt.Println(rotate180(d2r))

	fmt.Println(cnn(d1, d2[0]))
	fmt.Println(cnn(d1, d2r[0]))
	fmt.Println(matrixMul(d1, d2r))
}
