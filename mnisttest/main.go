package main

import (
	"cnn"
	"cnn/mnist"
	"fmt"
	"log"
)

func main() {
	dataTrain, err := mnist.ReadTrainSet("./mnist/MNIST_data")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("MNISST train: N:%v | W:%v | H:%v", dataTrain.N, dataTrain.W, dataTrain.H)

	c1 := cnn.NewConvolution(3, 3, 1, 32, 1)
	c1.LoadJSON("cnn_c1.json")
	settings := []interface{}{
		c1,
		cnn.NewMaxPool(2, 2),
	}
	softmax := cnn.NewSoftmax(14, 14, 32, 10)
	softmax.LoadJSON("cnn_softmax.json")
	learnRate := 0.9

	okla := make(map[int]bool)

	debug := false
	for count := 0; count < 1000; count++ {
		fmt.Println("\ncount:", count)
		right := 0
		for c, v := range dataTrain.Data {
			if _, ok := okla[c]; ok {
				continue
			}
			bits := cnn.Normalization(cnn.Uint8ToFloat64(v.Image))
			wants := make([]float64, 10)
			wants[v.Digit] = 1

			input := cnn.NewMatrix(28, 28, 1, nil)
			input.SetData([][][]float64{bits})
			out := input
			for _, vv := range settings {
				switch vv.(type) {
				case *cnn.Convolution:
					out = vv.(*cnn.Convolution).Forward(out)
					// print3(debug, "conv", out.GetData())
				case *cnn.MaxPool:
					out = vv.(*cnn.MaxPool).Forward(out)
					// print3(debug, "pool", out.GetData())
				}
			}

			outputs := softmax.Forward(out)
			// print1(debug, "outputs:", outputs)

			maxI, maxV := cnn.GetMax(outputs)
			if v.Digit == maxI && maxV > 0.9 {
				right++
				okla[c] = true
			}
			if c%100 == 0 {
				okr := float64(len(okla)) / float64(len(dataTrain.Data))
				fmt.Printf("\rRIGHT:[%.3f%%] %d %d %.2f (%d)", okr*100, c, v.Digit, outputs, len(dataTrain.Data)-len(okla))
				if okr > 0.8 {
					c1.SaveJSON("cnn_c1.json")
					softmax.SaveJSON("cnn_softmax.json")
					count = 1000000
					break
				}
			}

			if debug {
				fmt.Println("\n===== BACKPROP =====")
			}

			gradient := softmax.Backprop(wants, outputs, learnRate)
			// print3(debug, "softmax.backprop", gradient.GetData())

			for i := len(settings) - 1; i >= 0; i-- {
				switch settings[i].(type) {
				case *cnn.Convolution:
					// gradient = settings[i].(*Convolution).Backprop(gradient, learnRate)
					// print3(debug, "conv.filters", gradient.GetData())
				case *cnn.MaxPool:
					gradient = settings[i].(*cnn.MaxPool).Backprop(gradient)
					// print3(debug, "pool.backprop", gradient.GetData())
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
			bits := cnn.Normalization(cnn.Uint8ToFloat64(v.Image))
			wants := make([]float64, 10)
			wants[v.Digit] = 1

			input := cnn.NewMatrix(28, 28, 1, nil)
			input.SetData([][][]float64{bits})
			out := input
			for _, vv := range settings {
				switch vv.(type) {
				case *cnn.Convolution:
					out = vv.(*cnn.Convolution).Forward(out)
					// print3(debug, "conv", out.GetData())
				case *cnn.MaxPool:
					out = vv.(*cnn.MaxPool).Forward(out)
					// print3(debug, "pool", out.GetData())
				}
			}

			outputs := softmax.Forward(out)
			// print1(debug, "outputs:", outputs)

			maxI, maxV := cnn.GetMax(outputs)
			if v.Digit == maxI && maxV > 0.9 {
				right++
			}
			fmt.Printf("\rRIGHT:[%.3f%%] %d/%d", float64(right)/float64(len(dataCheck.Data))*100, c2, len(dataCheck.Data))
		}
	}
}
