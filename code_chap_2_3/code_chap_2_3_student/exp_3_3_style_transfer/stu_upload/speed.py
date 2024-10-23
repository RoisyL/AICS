    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time() 
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        self.height_out = (height - self.kernel_size) // self.stride + 1
        self.width_out = (width - self.kernel_size) // self.stride + 1
        self.weight_reshape = np.reshape(self.weight, [-1, self.channel_out]) # 对卷积核进行向量化
        self.img2col = np.zeros([self.input.shape[0]*self.height_out*self.width_out, self.channel_in*self.kernel_size*self.kernel_size])
        # 对卷积层的输入特征图进行向量化重排列
        for idxn in range(self.input.shape[0]):
            for idxh in range(self.height_out):
                for idxw in range(self.width_out):
                    self.img2col[idxn*self.height_out*self.width_out + idxh*self.width_out + idxw,:] = self.input_pad[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size].reshape([-1])
        # 计算卷积层的前向传播，特征图与卷积核的内积转变为矩阵相乘，再加偏置
        output = np.dot(self.img2col,self.weight_reshape) + self.bias
        self.output = output.reshape([self.input.shape[0], self.height_out, self.width_out, -1]).transpose([0, 3, 1, 2])    # 对卷积层的输出结果进行重排列
        self.forward_time = time.time() - start_time
        return self.output
    
    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        self.height_out = (height - self.kernel_size) // self.stride + 1
        self.width_out = (width - self.kernel_size) // self.stride + 1
        self.weight_reshape = np.reshape(self.weight, [-1, self.channel_out])
        self.img2col = np.zeros([self.input.shape[0]*self.height_out*self.width_out, self.channel_in*self.kernel_size*self.kernel_size])
        for idxn in range(self.input.shape[0]):
            for idxh in range(self.height_out):
                for idxw in range(self.width_out):
                    self.img2col[idxn*self.height_out*self.width_out + idxh*self.width_out + idxw, :] = self.input_pad[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size].reshape([-1])
        output = np.dot(self.img2col, self.weight_reshape) + self.bias
        self.output = output.reshape([self.input.shape[0], self.height_out, self.width_out, -1]).transpose([0, 3, 1, 2])
        self.forward_time = time.time() - start_time
        return self.output
    
    