// Copyright 2021 The Verse Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf32"
)

const (
	// Width is the width of the neural network
	Width = 8 * 8
)

func main() {
	rand.Seed(1)

	flag.Parse()

	set := tf32.NewSet()
	set.Add("aw1", Width, Width)
	set.Add("ab1", Width)
	set.Add("particles", Width, 8)

	for i := range set.Weights {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, 0)
			}
		} else {
			factor := float32(math.Sqrt(2 / float64(w.S[0])))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rand.NormFloat64())*factor)
			}
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	l1 := tf32.Softmax(tf32.Add(tf32.Mul(set.Get("aw1"), set.Get("particles")), set.Get("ab1")))
	cost := tf32.Avg(tf32.Quadratic(set.Get("particles"), l1))

	iterations, i, total := 30, 0, float32(0.0)
	alpha, eta := float32(.3), float32(.3)
	points := make(plotter.XYs, 0, iterations)
	for i < iterations {
		total = 0
		start := time.Now()
		set.Zero()

		total += tf32.Gradient(cost).X[0]
		norm := float32(0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = float32(math.Sqrt(float64(norm)))
		scaling := float32(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		for k, p := range set.Weights {
			for l, d := range p.D {
				deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
				p.X[l] += deltas[k][l]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total, time.Now().Sub(start))
		i++
	}

	err := set.Save("weights.w", total, i)
	if err != nil {
		panic(err)
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
}
