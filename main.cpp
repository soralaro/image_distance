#include <iostream>
#include <vector>
#include <fstream>
#include "opencv2/opencv.hpp"

int diffMatResizePlate(cv::Mat &src1, cv::Mat &src2, cv::Size srcSize, std::vector<int> &rst)
{
    float *p0 = (float*)src1.data;
    float *p1 = (float*)src2.data;

    unsigned int len = srcSize.height * srcSize.width;

    unsigned int k1=0, k2=0, k3=0, k4=0, k5=0, k6=0, k7=0;
    for (int j = 0; j < 3; j++){
        float *p00 = &p0[j * len];
        float *p11 = &p1[j * len];

        for (int i = 0; i < len; i++)
        {
            float diff = fabs(p00[i] - p11[i]);
            if (diff < 1e-6){
                k1++;
                continue;
            }
            else if (diff < 1e-5){
                k2++;
                continue;
            }
            else if (diff < 1e-4){
                k3++;
                continue;
            }
            else if (diff < 1e-3){
                k4++;
                continue;
            }
            else if (diff < 1e-2){
                k5++;
                continue;
            }
            else if (diff < 1e-1){
                k6++;
                continue;
            }
            else{ // diff >= 5
                k7++;
                continue;
            }
        }
    }

    rst.resize(7);
    rst[0] = k1;
    rst[1] = k2;
    rst[2] = k3;
    rst[3] = k4;
    rst[4] = k5;
    rst[5] = k6;
    rst[6] = k7;
    return 0;
}

double ssim(cv::Mat &i1, cv::Mat & mat2){
    const double C1 = 6.5025, C2 = 58.5225;
    cv::Mat i2(i1.size(), i1.type());
    i2.data = mat2.data;

    int d = CV_32F;
    cv::Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);
    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11,11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11,11), 1.5);
    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    cv::Mat sigma1_2, sigam2_2, sigam12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigam2_2, cv::Size(11, 11), 1.5);
    sigam2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigam12, cv::Size(11, 11), 1.5);
    sigam12 -= mu1_mu2;
    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigam12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigam2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    cv::Scalar mssim = mean(ssim_map);

    double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) /3;
    return ssim;
}
void getRGBvector(const cv::Mat&src, std::vector<unsigned int>& count)//得到64维向量
{
    int nRows = src.rows,nCols = src.cols * src.channels();
    const uchar* p;
    for (int i = 0; i < nRows; ++i)
    {
        p = src.ptr<uchar>(i);
        for (int j = 0; j < nCols; j += 3)
        {
            int r = int(p[j])/64;
            int g = int(p[j + 1])/64;
            int b = int(p[j + 2])/64;
            count[r * 16 + g * 4 + b]++;
        }
    }
}
double getVectorLength(std::vector<unsigned int> &vec)
{
    long long res = 0;
    for (int i = 0; i < vec.size(); i++)
        res += vec[i] * vec[i];
    return sqrt(res);
}

double getcos(std::vector<unsigned int> &count1, std::vector<unsigned int> &count2)
{
    double len1 = getVectorLength(count1);
    double len2 = getVectorLength(count2);
    assert(len1 != 0 && len2 != 0);
    long long sum = 0;
    for (int i = 0; i < count1.size(); i++)
        sum += count1[i] * count2[i];
    return (double)sum / len1 / len2 >0 ? (double)sum / len1 / len2:0;
}
double cossimilarity(const cv::Mat& src1, const cv::Mat& mat2) {
    std::vector<unsigned int> count1(64), count2(64);
    cv::Mat src2(src1.size(), src1.type());
    src2.data = mat2.data;

    getRGBvector(src1, count1);
    getRGBvector(src2, count2);
    double res = getcos(count1, count2);
    return res;
}
int main() {
    std::cout << "Hello, World!" << std::endl;
    float *pr_src=new float[8*108*108*18];
    std::ifstream inFile("/home/czx/work/Hi1910/vega_dir/vega/test/bin/face/binary.dat", std::ios::in | std::ios::binary);
    inFile.read((char*)pr_src,sizeof(float)*8*108*108*18);
    inFile.close();
    float *pr_dst=new float[8*108*108*18];
    std::ifstream inFile2("/home/czx/work/Hi1910/vega_dir/vega/vega/Debug/bin/binary_newcuda.dat", std::ios::in | std::ios::binary);
    inFile2.read((char*)pr_dst,sizeof(float)*8*108*108*18);
    inFile2.close();

    cv::Mat mat_src(108*3*6,108,CV_32FC1,pr_src);
    cv::Mat mat_dst(108*3*6,108,CV_32FC1,pr_dst);
    int c=0;
    for(int i=0;i<108*3*6;i++)
    {
        int src_a=pr_src[i]*100*100;
        int dst_a=pr_dst[i]*100;
        int dst_b=pr_dst[i]*10000;
        src_a=src_a-1;
        src_a/=100;

        if(dst_b*(-1)+dst_a*100>45)
        {
            dst_a-=1;

        }
        if(dst_a!=src_a)
        {
            printf("c=%d,i=%d ,%6f,%6f ,%d,%d,%d\n",c++,i, pr_dst[i],pr_src[i],dst_b,dst_a,src_a);
        }
    }
    double ret=ssim(mat_src,mat_dst);
    printf("ret=%lf\n",ret);

    delete [] pr_src;
    delete [] pr_dst;
    return 0;
}