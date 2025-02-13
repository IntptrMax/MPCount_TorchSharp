using System.Diagnostics;
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
			Device device = CUDA;
			ScalarType type = ScalarType.Float32;

			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

			var model = torch.jit.load(@".\Assets\model_fp16.torchscript").to(device, type);
			model.forward(torch.rand([1, 3, 1024, 1024]).to(type, device)); // Warm Up
			Stopwatch stopwatch = Stopwatch.StartNew();
			Tensor orgTensor = torchvision.io.read_image(@".\Assets\1.jpg").to(device);
			Tensor inputTensor = orgTensor.to(type).unsqueeze(0) / 255.0f / 2.0f - 1.0f;
			(Tensor, Tensor) resultTensors = ((Tensor, Tensor))(model.forward(inputTensor));
			Tensor r1 = resultTensors.Item1;

			int count = (int)(r1.sum() / 1000);
			var min = r1.min();
			var max = r1.max();
			Tensor resultMask = (r1.squeeze(0).squeeze(0) - min) / (max - min) > 0.2f;

			orgTensor[0] = (orgTensor[0] + resultMask * 255.0f).clamp(0, 255).@byte();

			MemoryStream memoryStream = new MemoryStream();
			torchvision.io.write_jpeg(orgTensor.cpu(), memoryStream);
			memoryStream.Position = 0;
			pictureBox1.Image = new Bitmap(memoryStream);

			stopwatch.Stop();
			textBox1.Text = $"Count: {count}\r\nTime: {stopwatch.ElapsedMilliseconds}ms";

		}
	}
}
