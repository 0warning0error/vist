use image::{DynamicImage, ImageOutputFormat, RgbaImage};
use image::imageops::{self, FilterType};
use serde::{Deserialize, Serialize};
use serde_cbor::{from_slice, to_vec};
use std::io::Cursor;
use wasm_minimal_protocol::*;
use image::Rgba;

initiate_protocol!();



#[wasm_func]
pub fn brighten(image_data_bytes: &[u8], brighten_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let value : i32 = from_slice(brighten_params_bytes).map_err(|e| e.to_string())?;
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let brightened_img = imageops::brighten(&img.to_rgba8(), value);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(brightened_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

#[wasm_func]
pub fn huerotate(image_data_bytes: &[u8], hue_rotate_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let value  = from_slice(hue_rotate_params_bytes).map_err(|e| e.to_string())?;
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let huerotated_img = imageops::huerotate(&img.to_rgba8(), value);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(huerotated_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

// Function to invert the image colors
#[wasm_func]
pub fn invert(image_data_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?.to_rgba8();
    imageops::invert(&mut img);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}


// Function to blur the image
#[wasm_func]
pub fn blur(image_data_bytes: &[u8], blur_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let sigma: f32 = from_slice(blur_params_bytes).map_err(|e| e.to_string())?;
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let blurred_img = imageops::blur(&img.to_rgba8(), sigma);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(blurred_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

// Function to apply an unsharpen mask to the image
#[wasm_func]
pub fn unsharpen(image_data_bytes: &[u8], unsharpen_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let (sigma, threshold): (f32, i32) = from_slice(unsharpen_params_bytes).map_err(|e| e.to_string())?;
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let unsharpened_img = imageops::unsharpen(&img.to_rgba8(), sigma, threshold);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(unsharpened_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}



// Function to adjust the contrast of the image
#[wasm_func]
pub fn adjust_contrast(image_data_bytes: &[u8], contrast_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let value = from_slice(contrast_params_bytes).map_err(|e| e.to_string())?;
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let contrasted_img = imageops::contrast(&img.to_rgba8(), value);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(contrasted_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

// Function to rotate the image 180 degrees
#[wasm_func]
pub fn rotate180(image_data_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let rotated_img = imageops::rotate180(&img.to_rgba8());
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(rotated_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

// Function to rotate the image 270 degrees
#[wasm_func]
pub fn rotate270(image_data_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let rotated_img = imageops::rotate270(&img.to_rgba8());
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(rotated_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}



#[wasm_func]
pub fn adjust_brightness(image_data_bytes: &[u8], brightness_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let value  = from_slice(brightness_params_bytes).map_err(|e| e.to_string())?;
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let brightened_img = imageops::brighten(&img.to_rgba8(), value);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(brightened_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}


// Function to convert the image to grayscale
#[wasm_func]
pub fn grayscale(image_data_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let gray_img = imageops::grayscale(&img);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(gray_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

#[derive(Serialize, Deserialize)]
struct CropParams {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

// Function to crop the image
#[wasm_func]
pub fn crop(image_data_bytes: &[u8], crop_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let CropParams { x, y, width, height } = from_slice(crop_params_bytes).map_err(|e| e.to_string())?;
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let sub_img = img.crop_imm(x, y, width, height);
    let mut bytes = Cursor::new(Vec::new());
    sub_img.write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

#[derive(Serialize, Deserialize)]
struct ThumbnailParams {
    width: u32,
    height: u32,
}

// Function to create a thumbnail of the image
#[wasm_func]
pub fn thumbnail(image_data_bytes: &[u8], thumbnail_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let ThumbnailParams { width, height } = from_slice(thumbnail_params_bytes).map_err(|e| e.to_string())?;
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let thumb_img = img.thumbnail(width, height);
    let mut bytes = Cursor::new(Vec::new());
    thumb_img.write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}
#[derive(Serialize, Deserialize)]
struct ResizeParams {
    width: u32,
    height: u32,
    filter_type: u8,
}

fn filter_type_from_u8(filter_type: u8) -> FilterType {
    match filter_type {
        0 => FilterType::Nearest,
        1 => FilterType::Triangle,
        2 => FilterType::CatmullRom,
        3 => FilterType::Gaussian,
        4 => FilterType::Lanczos3,
        _ => FilterType::Nearest, // 默认值
    }
}

// 修改后的resize函数，使用filter_type_from_u8函数
#[wasm_func]
pub fn resize(image_data_bytes: &[u8], resize_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let ResizeParams { width, height, filter_type } = from_slice(resize_params_bytes).map_err(|e| e.to_string())?;
    let filter = filter_type_from_u8(filter_type);
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let resized_img = imageops::resize(&img.to_rgba8(), width, height, filter);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(resized_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

#[derive(Serialize, Deserialize)]
struct ResizeToFillParams {
    width: u32,
    height: u32,
    filter_type: u8,
}

// Function to resize the image to fill specified dimensions, cropping if necessary
#[wasm_func]
pub fn resize_to_fill(image_data_bytes: &[u8], resize_to_fill_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let ResizeToFillParams { width, height, filter_type } = from_slice(resize_to_fill_params_bytes).map_err(|e| e.to_string())?;
    let filter = filter_type_from_u8(filter_type);
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let resized_img = img.resize_to_fill(width, height, filter);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(resized_img.into()).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

#[derive(Serialize, Deserialize)]
struct Filter3x3Params {
    kernel: [f32; 9],
}

// Function to apply a 3x3 filter to the image
#[wasm_func]
pub fn filter3x3(image_data_bytes: &[u8], filter3x3_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let kernel : [f32 ; 9] = from_slice(filter3x3_params_bytes).map_err(|e| e.to_string())?;
    
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let filtered_img = imageops::filter3x3(&img.to_rgba8(), &kernel);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(filtered_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

#[derive(Serialize, Deserialize)]
struct GradientParams {
    width: u32,
    height: u32,
    start_color: [u8; 4], // RGBA
    end_color: [u8; 4],   // RGBA
}

// Function to fill the image with a horizontal gradient
#[wasm_func]
pub fn horizontal_gradient(params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let GradientParams { width, height, start_color, end_color } = from_slice(params_bytes).map_err(|e| e.to_string())?;
    let mut image = RgbaImage::new(width, height);
    // 注意这里传递的是对Rgba值的引用
    imageops::horizontal_gradient(&mut image, &Rgba(start_color), &Rgba(end_color));
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    Ok(bytes.into_inner())
}

// Function to fill the image with a vertical gradient
#[wasm_func]
pub fn vertical_gradient(params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let GradientParams { width, height, start_color, end_color } = from_slice(params_bytes).map_err(|e| e.to_string())?;
    let mut image = RgbaImage::new(width, height);
    // 注意这里传递的是对Rgba值的引用
    imageops::vertical_gradient(&mut image, &Rgba(start_color), &Rgba(end_color));
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    Ok(bytes.into_inner())
}


#[derive(Serialize, Deserialize)]
struct OverlayParams {
    x: i64,
    y: i64,
}

// Function to overlay an image at a given coordinate (x, y)
#[wasm_func]
pub fn overlay( base_image_data_bytes: &[u8], overlay_image_data_bytes: &[u8],params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let OverlayParams { x, y } = from_slice(params_bytes).map_err(|e| e.to_string())?;
    let base_img = image::load_from_memory(base_image_data_bytes).map_err(|e| e.to_string())?;
    let overlay_img = image::load_from_memory(overlay_image_data_bytes).map_err(|e| e.to_string())?;
    
    let mut base_img_buf = base_img.to_rgba8();
    imageops::overlay(&mut base_img_buf, &overlay_img.to_rgba8(), x, y);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(base_img_buf).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    Ok(bytes.into_inner())
}

#[wasm_func]
pub fn tile( base_image_data_bytes: &[u8],image_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let base_img = image::load_from_memory(base_image_data_bytes).map_err(|e| e.to_string())?;
    let overlay_img = image::load_from_memory(image_bytes).map_err(|e| e.to_string())?;
    
    let mut base_img_buf = base_img.to_rgba8();
    imageops::tile(&mut base_img_buf, &overlay_img);
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(base_img_buf.into()).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    Ok(bytes.into_inner())
}

#[derive(Serialize, Deserialize)]
struct InterpolateParams {
    x: f32,
    y: f32,
}



// Function to interpolate a pixel value using bilinear sampling
#[wasm_func]
pub fn interpolate_bilinear(image_data_bytes: &[u8], params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    // Deserialize the image and parameters from the input bytes
    let image = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let params: InterpolateParams = from_slice(params_bytes).map_err(|e| e.to_string())?;
    
    // Perform bilinear interpolation
    let pixel = imageops::interpolate_bilinear(&image.to_rgba8(), params.x, params.y);
    
    // Serialize the result to bytes and return
    let result = pixel.unwrap().0;
    let result_bytes = to_vec(&result).map_err(|e| e.to_string())?;
    Ok(result_bytes)
}

// Similarly, you can wrap the `interpolate_nearest` function
#[wasm_func]
pub fn interpolate_nearest(image_data_bytes: &[u8], params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let image = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let params: InterpolateParams = from_slice(params_bytes).map_err(|e| e.to_string())?;

    // Perform nearest neighbor interpolation
    let pixel = imageops::interpolate_nearest(&image.to_rgba8(), params.x, params.y);

    // Serialize the result to bytes and return
    let result = pixel.unwrap().0;
    let result_bytes = to_vec(&result).map_err(|e| e.to_string())?;
    Ok(result_bytes)
}


// Function to flip the image vertically
#[wasm_func]
pub fn flip_vertical(image_data_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let flipped_img = imageops::flip_vertical(&img.to_rgba8());
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(flipped_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

// Function to flip the image horizontally
#[wasm_func]
pub fn flip_horizontal(image_data_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let flipped_img = imageops::flip_horizontal(&img.to_rgba8());
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(flipped_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

// Function to rotate the image 90 degrees clockwise
#[wasm_func]
pub fn rotate90(image_data_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    let rotated_img = imageops::rotate90(&img.to_rgba8());
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(rotated_img).write_to(&mut bytes, ImageOutputFormat::Png).map_err(|e| e.to_string())?;
    
    Ok(bytes.into_inner())
}

#[derive(Serialize, Deserialize)]
struct ConvertFormatParams {
    format: String, // 期望格式，如 "png", "jpeg" 等
    quality: Option<u8>, // 对于某些格式（如JPEG）的输出质量，如果适用
}

// Function to convert the image to a specified format
#[wasm_func]
pub fn convert_format(image_data_bytes: &[u8], format_params_bytes: &[u8]) -> Result<Vec<u8>, String> {
    // 解析格式和质量参数
    let ConvertFormatParams { format, quality } = from_slice(format_params_bytes).map_err(|e| e.to_string())?;
    
    // 加载图像
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    
    // 确定输出格式
    let output_format = match format.as_str() {
        "png" => ImageOutputFormat::Png,
        "jpeg" => ImageOutputFormat::Jpeg(quality.unwrap_or(80)), // 默认质量值为80
        "gif" => ImageOutputFormat::Gif,
        "ico" => ImageOutputFormat::Ico,
        "bmp" => ImageOutputFormat::Bmp,
        "farbfeld" => ImageOutputFormat::Farbfeld,
        "tga" => ImageOutputFormat::Tga,
        "openexr" => ImageOutputFormat::OpenExr,
        "tiff" => ImageOutputFormat::Tiff,
        "qoi" => ImageOutputFormat::Qoi,
        "webp" => ImageOutputFormat::WebP, // 默认质量值为80
        _ => return Err(format!("Unsupported image format: {}", format)),
    };
    
    // 转换图像
    let mut bytes = Cursor::new(Vec::new());
    img.write_to(&mut bytes, output_format).map_err(|e| e.to_string())?;
    
    // 返回转换后的字节流
    Ok(bytes.into_inner())
}

#[derive(Serialize, Deserialize)]
struct ImageMetaData {
    width: u32,
    height: u32,
    color_type: String,
    // format字段已移除
}

#[wasm_func]
pub fn get_image_metadata(image_data_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(image_data_bytes).map_err(|e| e.to_string())?;
    
    let meta_data = ImageMetaData {
        width: img.width(),
        height: img.height(),
        color_type: format!("{:?}", img.color()),
    };

    to_vec(&meta_data).map_err(|e| e.to_string())
}
