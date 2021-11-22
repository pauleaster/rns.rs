
use plotly::{ImageFormat, Layout, common::Title, contour::Contours, layout::Axis};
use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{Array2};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::error::Error;
use std::fs::File;
use plotly::{Contour, Plot};



pub fn write_array2(array: &mut Array2<f64>, filename: &str ) -> Result<(), Box<dyn Error>> {

    let file = File::create(filename)?;
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
    writer.serialize_array2(array)?;
    
    
    Ok(())

}

#[allow(clippy::many_single_char_names)]
pub fn contourf(s: &[f64], m: &[f64], array: &Array2<f64>, 
                name: &str, filename: &str, 
                tag: &str, postfix_filename: &str) {

    let mut x: Vec<f64> = vec![];
    let mut y: Vec<f64> = vec![];
    let mut z: Vec<f64> = vec![];

    // println!(array.);
    for ((i,j), &zv) in array.indexed_iter() {
        // print!("({},{}) ",i,j);
        x.push(s[i]);
        y.push(m[j]);
        z.push(zv);

    }
    let trace = Contour::new(x,y,z);
    let mut plot = Plot::new();
    plot.add_trace(trace);

    let layout = Layout::new()
                        .title(Title::new(&format!("{}{}",name,tag)))
                        .x_axis(Axis::new().title(Title::new("s coordinate")))
                        .y_axis(Axis::new().title(Title::new("angular coordinate, cos θ")));
    plot.set_layout(layout);
    // plot.show_png(800,600);
    // plot.to_html(&format!("{}{}{}",filename, postfix_filename, ".html"));
    let relative_filename = format!("{}{}{}",filename, postfix_filename, ".png");
    plot.save(relative_filename,ImageFormat::PNG,1280,900,1.0);

    
}