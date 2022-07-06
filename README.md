# End To End Compression

Our team's work on USTC iVC Seminar programming playground. 

## Baseline

We use the framework proposed in the following paper as our work's baseline.

Ballé J, Laparra V, Simoncelli E P. End-to-end optimized image compression[J]. arXiv preprint arXiv:1611.01704, 2016.

The paper is available [here](https://arxiv.org/pdf/1611.01704.pdf).

## Framework

### GDN

```python
class GDN(nn.Module):
    def __init__(self, num_output_channel, beta_min=1e-6, beta_init=0.1, gamma_min=1e-6, gamma_init=0.1,
                 min_boundary=2e-5, inverse=False):
        """
        :param beta_min: a small positive value to ensure beta' in range(2e-5,...)
        :param gamma_init: gamma initiated value
        :param num_output_channel: It is same for in/out because it is only a 'nomalization'
        :param min_boundary: the lower boundary for 'gamma' and 'beta''
        :param inverse: Identify GDN or IGDN
        """
        super(GDN, self).__init__()
        self.min_boundary = min_boundary
        self.inverse = inverse
        self.num_output_channel = num_output_channel
        self.reparam_offset = min_boundary ** 2
        self.beta_bound = (beta_min + self.reparam_offset) ** 0.5
        self.gamma_bound = (gamma_min + self.reparam_offset) ** 0.5

        # beta, gamma
        self.beta = nn.Parameter(torch.sqrt(torch.ones(num_output_channel) * beta_init + self.reparam_offset))
        self.gamma = nn.Parameter(torch.sqrt(torch.eye(num_output_channel) * gamma_init + self.reparam_offset))

    def forward(self, inputs):
        # transpose average
        gamma_T = self.gamma.transpose(0, 1)
        gamma_p = (self.gamma + gamma_T) / 2

        # lower boundary
        beta_p = SetMinBoundary.apply(self.beta, self.beta_bound)
        beta = beta_p ** 2 - self.reparam_offset

        gamma_p = SetMinBoundary.apply(gamma_p, self.gamma_bound)
        gamma = gamma_p ** 2 - self.reparam_offset
        # tensor转化为一维
        gamma = gamma.view(self.num_output_channel, self.num_output_channel, 1, 1)

        # normalization, resemble to 2d conv with kernel size set to 1
        norm = F.conv2d(inputs ** 2, gamma,
                        beta)  # 采用二维卷积来实现[batch_size, channel_size, H, W]*[channel_size, channel_size, 1 ,1 ]
        if self.inverse:
            outputs = inputs * norm
        else:
            outputs = inputs / norm
        return outputs
```

## Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
