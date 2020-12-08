package cnn

import (
	"cnn/mnist"
	"fmt"
	"log"
	"testing"
)

//*
// go test cnn -run TestCNN -v -count=1
func TestCNN(t *testing.T) {
	input := NewMatrix(3, 3, 2, func() float64 { return Random(0, 1) })
	// input.SetData([][][]float64{Normalization([][]float64{
	// 	{1, 2, 3},
	// 	{4, 5, 6},
	// 	{7, 8, 9},
	// }), Normalization([][]float64{
	// 	{11, 22, 33},
	// 	{44, 55, 66},
	// 	{77, 88, 99},
	// })})
	wants := []float64{1, 0}

	c := NewConvolution(3, 3, 2, 2, 1)
	// c.filters[0].SetData([][][]float64{
	// 	{
	// 		{1, 0, 1},
	// 		{0, 1, 0},
	// 		{1, 0, 1},
	// 	},
	// 	{
	// 		{0, 1, 0},
	// 		{1, 0, 1},
	// 		{0, 1, 0},
	// 	},
	// })
	settings := []interface{}{
		// 5x5x2
		c,
		// 5x5x2
		// NewConvolution(3, 3, 2, 2, 1),
		// 5x5x2
		NewMaxPool(2, 2),
		// 3x3x2
		// NewConvolution(3, 3, 2, 2, 1),
		// 3x3x2
		// NewMaxPool(2, 2),
		// 2x2x2
	}
	softmax := NewSoftmax(2, 2, 2, len(wants))
	// softmax := NewSoftmax(3, 3, 2, len(wants))

	learnRate := 0.6
	count := 1
	debug := false
	if count == 1 {
		debug = true
		fmt.Print("\033c")
	}

	// iw, ih, id := len(input[0][0]), len(input[0]), len(input)
	// sw, sh := SoftmaxHelper(iw, ih, conv, pool)
	// fmt.Printf("input size: %d*%d*%d\n", iw, ih, id)
	// fmt.Printf("conv size: %d*%d*%d step:%d\n", conv.w, conv.h, conv.depth, conv.step)
	// fmt.Printf("pool size: %d*%d\n", pool.w, pool.h)
	// fmt.Printf("softmax size: %d*%d\n", sw, sh)
	// fmt.Printf("learn rate: %.4f\n", learnRate)
	// softmax := NewSoftmax(sw, sh, conv.depth*len(input), len(wants))

	// conv[2x2] => conv[2x2] => pool[2x2] => conv[2x2] => pool[2x2] => softmax
	// 5x5          5x5          3x3          3x3          2x2          4
	for c := 0; c < count; c++ {
		out := input
		for _, v := range settings {
			switch v.(type) {
			case *Convolution:
				out = v.(*Convolution).Forward(out)
				print3(debug, "conv", out.GetData())
			case *MaxPool:
				out = v.(*MaxPool).Forward(out)
				print3(debug, "pool", out.GetData())
			}
		}

		outputs := softmax.Forward(out)
		print1(debug, "outputs:", outputs)

		if count == 1 || c%(count/10) == 0 {
			fmt.Printf("LOSS: %d %.8f, %.8f\n", c, wants, outputs)
		}

		if debug {
			fmt.Println("\n===== BACKPROP =====")
		}

		gradient := softmax.Backprop(wants, outputs, learnRate)
		print3(debug, "softmax.backprop", gradient.GetData())

		for i := len(settings) - 1; i >= 0; i-- {
			switch settings[i].(type) {
			case *Convolution:
				gradient = settings[i].(*Convolution).Backprop(gradient, learnRate)
				print3(debug, "conv.backprop", gradient.GetData())
			case *MaxPool:
				gradient = settings[i].(*MaxPool).Backprop(gradient)
				print3(debug, "pool.backprop", gradient.GetData())
			}
		}
	}
}

//*/

//*
// go test cnn -run TestMnist -v -count=1
func TestMnist(t *testing.T) {
	dataTrain, err := mnist.ReadTrainSet("./mnist/MNIST_data")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("MNISST train: N:%v | W:%v | H:%v", dataTrain.N, dataTrain.W, dataTrain.H)

	settings := []interface{}{
		NewConvolution(3, 3, 1, 32, 1),
		// NewConvolution(3, 3, 2, 2, 1),
		NewMaxPool(2, 2),
		// NewMaxPool(2, 2),
	}
	softmax := NewSoftmax(14, 14, 32, 10)
	learnRate := 0.6

	okla := make(map[int]bool)

	debug := false
	for count := 0; count < 1000; count++ {
		fmt.Println("\ncount:", count)
		right := 0
		for c, v := range dataTrain.Data {
			if _, ok := okla[c]; ok {
				continue
			}
			bits := Normalization(Uint8ToFloat64(v.Image))
			wants := make([]float64, 10)
			wants[v.Digit] = 1

			input := NewMatrix(28, 28, 1, nil)
			input.SetData([][][]float64{bits})
			out := input
			for _, vv := range settings {
				switch vv.(type) {
				case *Convolution:
					out = vv.(*Convolution).Forward(out)
					print3(debug, "conv", out.GetData())
				case *MaxPool:
					out = vv.(*MaxPool).Forward(out)
					print3(debug, "pool", out.GetData())
				}
			}

			outputs := softmax.Forward(out)
			print1(debug, "outputs:", outputs)

			maxI, maxV := GetMax(outputs)
			if v.Digit == maxI && maxV > 0.9 {
				right++
				okla[c] = true
			}
			if c%100 == 0 {
				okr := float64(len(okla)) / float64(len(dataTrain.Data))
				fmt.Printf("\rRIGHT:[%.3f%%] %d %d %.2f (%d)", okr*100, c, v.Digit, outputs, len(dataTrain.Data)-len(okla))
				if okr > 0.8 {
					break
				}
			}

			if debug {
				fmt.Println("\n===== BACKPROP =====")
			}

			gradient := softmax.Backprop(wants, outputs, learnRate)
			print3(debug, "softmax.backprop", gradient.GetData())

			for i := len(settings) - 1; i >= 0; i-- {
				switch settings[i].(type) {
				case *Convolution:
					// gradient = settings[i].(*Convolution).Backprop(gradient, learnRate)
					print3(debug, "conv.filters", gradient.GetData())
				case *MaxPool:
					gradient = settings[i].(*MaxPool).Backprop(gradient)
					print3(debug, "pool.backprop", gradient.GetData())
				}
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
			bits := Normalization(Uint8ToFloat64(v.Image))
			wants := make([]float64, 10)
			wants[v.Digit] = 1

			input := NewMatrix(28, 28, 1, nil)
			input.SetData([][][]float64{bits})
			out := input
			for _, vv := range settings {
				switch vv.(type) {
				case *Convolution:
					out = vv.(*Convolution).Forward(out)
					print3(debug, "conv", out.GetData())
				case *MaxPool:
					out = vv.(*MaxPool).Forward(out)
					print3(debug, "pool", out.GetData())
				}
			}

			outputs := softmax.Forward(out)
			print1(debug, "outputs:", outputs)

			maxI, maxV := GetMax(outputs)
			if v.Digit == maxI && maxV > 0.9 {
				right++
			}
			fmt.Printf("\rLOSS:[%.3f%%]", float64(c2-right)/float64(c2)*100)
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

	d1 := NewMatrix(5, 5, 1, nil)
	d2 := NewMatrix(4, 4, 1, nil)

	d1.SetData([][][]float64{
		{
			{0.51, 0.9, 0.88, 0.84, 0.05},
			{0.4, 0.62, 0.22, 0.59, 0.1},
			{0.11, 0.2, 0.74, 0.33, 0.14},
			{0.47, 0.01, 0.85, 0.7, 0.09},
			{0.76, 0.19, 0.72, 0.17, 0.57},
		},
	})
	d2.SetData([][][]float64{
		{
			{0, 0, 0.0686, 0},
			{0, 0.0364, 0, 0},
			{0, 0.0467, 0, 0},
			{0, 0, 0, -0.0681},
		},
	})
	fmt.Println(cnn(d1.GetData()[0], d2.GetData()[0]))

	d2.Rotate180()
	fmt.Println(d2.GetData())
	// fmt.Println(d1.Conv(d2).GetData())
}

//*/
// go test -run Test_matrixMul cnn -v -count=1
// func Test_matrixMul(t *testing.T) {
// 	input := [][][]float64{
// 		{
// 			{1, 2, 3, 2, 1},
// 			{4, 5, 6, 5, 4},
// 			{7, 8, 9, 8, 7},
// 			{4, 5, 6, 5, 4},
// 			{1, 2, 3, 2, 1},
// 		},
// 		{
// 			{11, 22, 33, 22, 11},
// 			{44, 55, 66, 55, 44},
// 			{77, 88, 99, 88, 77},
// 			{44, 55, 66, 55, 44},
// 			{11, 22, 33, 22, 11},
// 		},
// 	}

// 	filter := [][][]float64{
// 		{
// 			{0.1, 0.2, 0.3},
// 			{0.4, 0.5, 0.6},
// 			{0.7, 0.8, 0.9},
// 		},
// 		{
// 			{0.11, 0.22, 0.33},
// 			{0.44, 0.55, 0.66},
// 			{0.77, 0.88, 0.99},
// 		},
// 	}

// 	out := matrixMul(input, filter)
// 	print3(true, "out", out)
// }
