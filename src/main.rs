use std::path::Path;
use num::Zero;
use video_rs::encode::{Encoder, Settings};
use video_rs::time::Time;
use ndarray::Array3;
use num::complex::Complex;

const WIDTH: usize = 384;
const HEIGHT: usize = 216;
const FPS: usize = 24;

const MASS: f32 = 1.;

fn potential(_x: usize, _y: usize) -> f32 {
    0.
}

fn main() {
    // set up video generation
    video_rs::init().unwrap();
    let settings = Settings::preset_h264_yuv420p(WIDTH, HEIGHT, false);
    let mut encoder = Encoder::new(Path::new("output.mp4"), settings)
        .expect("encoding error");

    // set up simulation state as 2d array of complex values
    let a: f32 = 0.1;
    let mut state: [[Complex<f32>; HEIGHT]; WIDTH] = [[Complex::zero();HEIGHT];WIDTH];

    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            state[x][y] = 
                wavepacket(x as f32 - 50., y as f32, a);
                //+ wavepacket(x as f32 + 50. as f32, y as f32, a)
                //+ wavepacket(x as f32, y as f32 - 43.3, a);
        }
    }

    normalize(&mut state);

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

        for n in 0..10 {
            state = sim_step(state, 0.1);
            if n % 10 == 0 {normalize(&mut state);}
        }
    }
    println!("finished last frame");

    encoder.finish().expect("encoding error");
}

fn framegen(state: &[[Complex<f32>; HEIGHT]; WIDTH]) -> Array3<u8> {
    Array3::from_shape_fn((HEIGHT,WIDTH,3), |(x,y,c)| 
        complex_visualizer(&state[y][x])[c]
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
    let v = (c.norm_sqr() / 25.).tanh() * 255.;

    [(r*v) as u8, (g*v) as u8, 0]
}

// actual physics fr fr

fn sim_step(state: [[Complex<f32>; HEIGHT]; WIDTH], dt: f32) -> [[Complex<f32>; HEIGHT]; WIDTH] {
    let mut substep_state = [[Complex::ZERO; HEIGHT]; WIDTH];

    let mut k1: [[Complex<f32>; HEIGHT]; WIDTH] = [[Complex::ZERO; HEIGHT]; WIDTH];
    let mut k2: [[Complex<f32>; HEIGHT]; WIDTH] = [[Complex::ZERO; HEIGHT]; WIDTH];
    let mut k3: [[Complex<f32>; HEIGHT]; WIDTH] = [[Complex::ZERO; HEIGHT]; WIDTH];
    // not needed, integration done in last step with k4 generation
    //let mut k4: [[Complex<f32>; HEIGHT]; WIDTH] = [[Complex::ZERO; HEIGHT]; WIDTH];

    // calculate k1 and prepare for k2 calculation
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            k1[x][y] = Complex::<f32>::i() * (0.5 / MASS * discrete_laplace_operator(&state, x, y) - potential(x, y) * state[x][y]);
            substep_state[x][y] = state[x][y] + dt * k1[x][y] * 0.5;
        }
    }
    // calculate k2
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            k2[x][y] = Complex::<f32>::i() * (0.5 / MASS * discrete_laplace_operator(&substep_state, x, y) - potential(x, y) * state[x][y]);
        }
    }
    // prepare for k3 calculation
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            substep_state[x][y] = state[x][y] + dt * k2[x][y] * 0.5;
        }
    }
    // calculate k3
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            k3[x][y] = Complex::<f32>::i() * (0.5 / MASS * discrete_laplace_operator(&substep_state, x, y) - potential(x, y) * state[x][y]);
        }
    }
    // prepare for k4 calculation
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            substep_state[x][y] = state[x][y] + dt * k3[x][y];
        }
    }
    // calculate k4
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            let k4 = Complex::<f32>::i() * (0.5 / MASS * discrete_laplace_operator(&substep_state, x, y) - potential(x, y) * state[x][y]);
            substep_state[x][y] = state[x][y] + dt / 6. * (k1[x][y] + 2. * k2[x][y] + 2. * k3[x][y] + k4);
        }
    }

    substep_state
}

fn discrete_laplace_operator(field: &[[Complex<f32>; HEIGHT]; WIDTH], x: usize, y: usize) -> Complex<f32> {
    unsafe {(1./6.) * (
        field.get_unchecked(x.unchecked_sub(1)).get_unchecked(y.unchecked_sub(1))
        + 4. * field.get_unchecked(x.unchecked_sub(1)).get_unchecked(y)
        + field.get_unchecked(x.unchecked_sub(1)).get_unchecked(y.unchecked_add(1))
        + 4. * field.get_unchecked(x).get_unchecked(y.unchecked_sub(1))
        - 20. * field.get_unchecked(x).get_unchecked(y)
        + 4. * field.get_unchecked(x).get_unchecked(y.unchecked_add(1))
        + field.get_unchecked(x.unchecked_add(1)).get_unchecked(y.unchecked_sub(1))
        + 4. * field.get_unchecked(x.unchecked_add(1)).get_unchecked(y)
        + field.get_unchecked(x.unchecked_add(1)).get_unchecked(y.unchecked_add(1))
    )}
}

fn normalize(state: &mut [[Complex<f32>; HEIGHT]; WIDTH]) {
    let v = state.iter().flat_map(|a| a.iter()).map(|n| n.norm_sqr()).sum::<f32>();
    state.iter_mut().for_each(|a| a.iter_mut().for_each(|n| *n *= ((WIDTH * HEIGHT) as f32 / v).sqrt()));
}

fn wavepacket(x: f32, y: f32, a: f32) -> Complex<f32> {
    let mag = f32::exp(-a.powi(2) * ((x - WIDTH as f32 / 2.).powi(2) + (y - HEIGHT as f32/ 2.).powi(2)));
    Complex::from_polar(mag, -x * 0.5)
}