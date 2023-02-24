#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <string> 
#include <matplot/matplot.h>

using namespace matplot;


//Convert a tensor 1-D into a std::vector of double
std::vector<double> to_vector(torch::Tensor tensor){

    std::vector<float> v(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    std::vector<double> vec ;
    for(auto elem : v ){
        vec.push_back((double)elem);
    }
    return vec;
}

//Our network
struct LRImpl : torch::nn::Module {
  LRImpl(int64_t inS, int64_t outS):
  linear(register_module("linear",torch::nn::Linear(inS,outS)))
  {

  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input);
  }
  torch::nn::Linear linear;
};

TORCH_MODULE(LR);

int main() {

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    //Data samples
    auto x_data = torch::randn({100,1})*10;
    auto y_data = x_data + 3 * torch::randn({100,1});

    /*
    *For plotting our data we will have to convert them into std::vector<double>
    */
    auto x_data_vector = to_vector(x_data);
    auto y_data_vector = to_vector(y_data);

    //We can now plot them
    matplot::scatter(x_data_vector, y_data_vector);
    matplot::hold(on);
    matplot::save("IMAGE_data/data_before.png");

    // Hyper parameters
    const int64_t input_size = 1;
    const int64_t output_size = 1;
    const size_t num_epochs = 100;
    const double learning_rate = 0.01;


    // Linear regression model
    LR model(input_size, output_size);
    model->to(device);

    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(8);

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Forward pass
        auto output = model->forward(x_data);
        auto loss = torch::nn::functional::mse_loss(output, y_data);

        // Backward pass and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        
        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs <<
                "], Loss: " << loss.item<double>() << "\n";
                
    }

    std::vector<float> xx1 = {-30, 30};
    torch::Tensor X = torch::from_blob(xx1.data(), {2, 1});

    auto A = model->parameters()[0];
    auto B = model->parameters()[1];

    double aa = to_vector(A)[0];
    double bb = to_vector(B)[0];

    auto Y = aa * X + bb;

    auto yy = to_vector(Y); 
    auto xx = to_vector(X);
    matplot::plot(xx, yy);
    matplot::save("IMAGE_data/end.png");
    matplot::hold(off);
}