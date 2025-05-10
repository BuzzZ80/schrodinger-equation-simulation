use std::path::Path;
use num::Zero;
use video_rs::encode::{Encoder, Settings};
use video_rs::time::Time;
use ndarray::Array3;
use num::complex::Complex;

const WIDTH: usize = 128;
const HEIGHT: usize = 128;
const FPS: usize = 24;

const MASS: f32 = 1.;

fn potential(x: f32, y: f32) -> f32 {
    let x = x - WIDTH as f32;
    let y = y - HEIGHT as f32;

    let r = (x*x + y*y).sqrt();
    if r < 1. {
        0.
    } else {
        -100. / r
    }
}

fn main() {
    // set up video generation
    video_rs::init().unwrap();
    let settings = Settings::preset_h264_yuv420p(WIDTH, HEIGHT, false);
    let mut encoder = Encoder::new(Path::new("output.mp4"), settings)
        .expect("encoding error");

    // set up simulation state as 2d array of complex values
    let d = 30.;
    let a: f32 = 0.3;
    let mut state: [[Complex<f32>; HEIGHT]; WIDTH] = [[Complex::zero();HEIGHT];WIDTH];
    
    // doubly-localized
    /*for x in 0..WIDTH {
        for y in 0..HEIGHT {
            state[x][y] = 
                (f32::exp(-a.powi(2) * ((x as f32 - WIDTH as f32 / 2. - d).powi(2) as f32 + (y as f32 - 138.).powi(2) as f32))
                + f32::exp(-a.powi(2) * ((x as f32 - WIDTH as f32 / 2. + d).powi(2) as f32 + (y as f32 - 118.).powi(2) as f32))).into()
        }
    }*/

    //singly-localized
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            state[x][y] = 
                (f32::exp(-a.powi(2) * ((x as f32 - WIDTH as f32 / 2. - d).powi(2) as f32 + (y as f32 - HEIGHT as f32/ 2. - d).powi(2) as f32))).into()
        }
    }

    // really just for normalization
    sim_step(state, 0.001);

    // step through simulation while making video
    let frametime = Time::from_nth_of_a_second(FPS);
    let mut current_time = Time::zero();

    let num_frames = FPS * 20;
    let start = std::time::Instant::now();
    for i in 0..num_frames {
        println!(
            "generating frame {i} of {num_frames} ({}%) - elapsed {:?} est. {:?} remaining", 
            (i * 100) / num_frames, 
            start.elapsed().as_secs(),
            ((start.elapsed().as_secs_f32() / i as f32) * (num_frames - i) as f32) as u32
        );

        let frame = framegen(&state);
        encoder.encode(&frame, current_time)
            .expect("encoding error");
        current_time = current_time.aligned_with(frametime).add();

        for _ in 0..10000 {
            state = sim_step(state, 0.00001);
        }
    }
    println!("finished last frame");

    encoder.finish().expect("encoding error");
}

fn framegen(state: &[[Complex<f32>; HEIGHT]; WIDTH]) -> Array3<u8> {
    Array3::from_shape_fn((HEIGHT,WIDTH,3), |(x,y,c)| 
        complex_visualizer(&state[x][y])[c]
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

fn sim_step(state: [[Complex<f32>; HEIGHT]; WIDTH], dt: f32) -> [[Complex<f32>; HEIGHT]; WIDTH] {
    let mut newstate = state.clone();
    let mut normalizing_factor: f32 = 0.;

    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            let nabla_sq =
                (state[x-1][y-1] + 4.*state[x][y-1] + state[x+1][y-1]
                + 4.*state[x-1][y] - 20.*state[x][y] + 4.*state[x+1][y]
                + state[x-1][y+1] + 4.*state[x][y+1] + state[x+1][y+1]) / 6.;

            newstate[x][y] += dt * Complex::i() * (0.5 / MASS * nabla_sq + potential(x as f32,y as f32) * state[x][y]);
            normalizing_factor += state[x][y].norm_sqr();
        }
    }

    let scale = ((WIDTH * HEIGHT) as f32 / normalizing_factor).sqrt();
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            newstate[x][y] *= scale;
        }
    }

    newstate
}