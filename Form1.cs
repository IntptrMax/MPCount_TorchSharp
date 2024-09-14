using System.Diagnostics;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace MPCount_TorchSharp
{
	public partial class Form1 : Form
	{
		public Form1()
		{
			InitializeComponent();
		}

		private void Form1_Load(object sender, EventArgs e)
		{
			var model = torch.jit.load(@".\Assets\model_fp16.torchscript").half().cuda();
			model.forward(torch.rand([1, 3, 1024, 1024]).half().cuda()); // Warm Up
			Bitmap srcBitmap = new Bitmap(@".\Assets\1.jpg");
			int width = srcBitmap.Width;
			int height = srcBitmap.Height;
			BitmapData bitmapData = srcBitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
			byte[] data = new byte[bitmapData.Stride * bitmapData.Height];
			Marshal.Copy(bitmapData.Scan0, data, 0, data.Length);
			srcBitmap.UnlockBits(bitmapData);

			float[] tensorData = new float[data.Length];

			for (int c = 0; c < 3; c++)
			{
				for (int w = 0; w < width; w++)
				{
					for (int h = 0; h < height; h++)
					{
						tensorData[c * (width * height) + (width * h + w)] = (data[3 * width * h + 3 * w + 2 - c] / 255.0f - 0.5f) / 0.5f;
					}
				}
			}

			Tensor inputTensor = torch.tensor(tensorData).half().cuda();
			inputTensor = inputTensor.view([1, 3, height, width]);
			Stopwatch stopwatch = Stopwatch.StartNew();
			ValueTuple<Tensor, Tensor> resultTensors = (ValueTuple<Tensor, Tensor>)model.forward(inputTensor);

			Tensor r1 = resultTensors.Item1;
			//Tensor r2 = resultTensors.Item2;

			float[] f = r1.@float().data<float>().ToArray();
			int count = (int)(f.Sum() / 1000);
			var min = f.Min();
			var max = f.Max();
			Tensor result = (r1 - min) / (max - min) * 255.0f;

			byte[] resultData = new byte[f.Length];
			for (int i = 0; i < resultData.Length; i++)
			{
				float r = (f[i] - min) / (max - min);
				if (r > 0.5)
				{
					data[3 * i + 2] = 255;
				}
			}

			Bitmap predImg = new Bitmap(width, height, PixelFormat.Format24bppRgb);
			BitmapData predBitmapData = predImg.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
			Marshal.Copy(data, 0, predBitmapData.Scan0, data.Length);
			predImg.UnlockBits(predBitmapData);
			stopwatch.Stop();
			textBox1.Text = "Time£º" + stopwatch.ElapsedMilliseconds + " ms\r\n" + "Count£º" + count;
			pictureBox1.Image = predImg;

		}
	}
}
