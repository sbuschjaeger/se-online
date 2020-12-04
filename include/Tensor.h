template<typename data_t>

class Tensor {

public:
    unsigned int x_max;
    unsigned int y_max;
    unsigned int z_max;
    data_t * data = nullptr;

    Tensor(unsigned int dim1, unsigned dim2, unsigned int dim3) : dim1(x_max), dim2(y_max), dim3(z_max) {
        data = new data_t[dim1*dim2*dim3];
    }

    data_t & operator()(unsigned int x, unsigned int y, unsigned int z) {
        return data[(z * x_max * y_max) + (y * x_max) + x]; //x + WIDTH * (y + DEPTH * z)
    }

    ~Tensor() {
        if (data != nullptr) {
            delete[] data;
        }
    }

};