use std::path::Path;
use num::{One, Zero};
use video_rs::encode::{Encoder, Settings};
use video_rs::time::Time;
use ndarray::{Array2, Array3, ArrayView2, IndexLonger};
use num::complex::Complex;

const WIDTH: usize = 256;
const HEIGHT: usize = 256;
const FPS: usize = 24;

const MASS: f32 = 2.;

fn main() {
    // set up video generation
    video_rs::init().unwrap();
    let settings = Settings::preset_h264_yuv420p(WIDTH, HEIGHT, false);
    let mut encoder = Encoder::new(Path::new("output.mp4"), settings)
        .expect("encoding error");

    // set up simulation state as 2d array of complex values
    let d = 30;
    let a: f32 = 0.1;
    let mut state = Array2::<Complex<f32>>::from_shape_fn((HEIGHT, WIDTH), |(y,x)| {
        (f32::exp(-a.powi(2) * ((x-WIDTH/2-d).pow(2) as f32 + (y as f32 - 20.).powi(2) as f32))
        + f32::exp(-a.powi(2) * ((x-WIDTH/2+d).pow(2) as f32 + (y as f32 - 20.).powi(2) as f32))).into()
    });

    // step through simulation while making video
    let frametime = Time::from_nth_of_a_second(FPS);
    let mut current_time = Time::zero();

    let num_frames = FPS * 30;
    for i in 0..num_frames {
        println!("generating frame {i} of {num_frames} ({}%)", (i * 100) / num_frames);
        for _ in 0..4000 {
            state = sim_step(state, 0.001);
        }

        let frame = framegen(state.view());
        encoder.encode(&frame, current_time)
            .expect("encoding error");
        current_time = current_time.aligned_with(frametime).add();
    }
    println!("finished last frame");

    encoder.finish().expect("encoding error");
}

fn framegen(state: ArrayView2<Complex<f32>>) -> Array3<u8> {
    Array3::from_shape_fn((HEIGHT,WIDTH,3), |(x,y,c)| 
        complex_visualizer(state.index((x,y)))[c]
    )
}

fn complex_visualizer(c: &Complex<f32>) -> [u8; 3] {
    // probability visualization
    /*
    let v = (c.norm_sqr() * 256.) as u8;
    [v,v,v]
    */

    // actual complex value visualization
    let r = c.arg().cos().powi(2);
    let g = c.arg().sin().powi(2);
    let v = c.norm_sqr().tanh() * 255.;

    [(r*v) as u8, (g*v) as u8, 0]
}

// actual physics fr fr

fn sim_step(state: Array2<Complex<f32>>, dt: f32) -> Array2<Complex<f32>> {
    let mut newstate: Array2<Complex<f32>> = state.clone();
    let mut normalizing_factor: f32 = 0.;

    (0..WIDTH).map(|x| std::thread::spawn(|| {

    })).for_each(|handle| handle.join().unwrap());

    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let dest = newstate.get_mut((y,x)).unwrap();
            let epdx = *state.get((y,x+1)).unwrap_or(&Complex::zero());
            let emdx = *state.get((y,x.wrapping_sub(1))).unwrap_or(&Complex::zero());
            let epdy = *state.get((y+1,x)).unwrap_or(&Complex::zero());
            let epmy = *state.get((y.wrapping_sub(1),x)).unwrap_or(&Complex::zero());

            let d2dx2 = epdx - 2. * *dest + emdx;
            let d2dy2 = epdy - 2. * *dest + epmy;
            let nabla_sq = d2dx2 + d2dy2;

            *dest += dt * Complex::i() * 0.5 / MASS * nabla_sq;
            normalizing_factor += dest.norm_sqr();
        }
    }


    let scale = ((WIDTH * HEIGHT) as f32 / normalizing_factor).sqrt();
    newstate.map(|n| n * scale)
}