//
// Create by Adarion on 2022/07/30
//

#pragma once

#ifndef TINY_SDF_HPP
#define TINY_SDF_HPP

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

cv::Mat _8SSEDT(const cv::Mat& img)
{
    struct _8SSEDT_struct
    {
        int x, y;

        inline int getDistance() const
        {
            return x * x + y * y;
        }

        inline void toMinVal(const _8SSEDT_struct& a)
        {
            if (a.getDistance() < getDistance())
                x = a.x, y = a.y;
        }

        inline _8SSEDT_struct operator+(const _8SSEDT_struct& a) const
        {
            return {x + a.x, y + a.y};
        }
    };

    static constexpr int infinity = 9999;
    static constexpr int offsetX[] = {-1, -1, 0, 1, 1, 1, 0, -1};
    static constexpr int offsetY[] = {0, -1, -1, -1, 1, 0, 1, 1};

    int row = img.rows, col = img.cols;

    //vector<vector<_8SSEDT_struct>> gridB(row + 2, vector<_8SSEDT_struct>(col + 2, {0, 0})),
    //                    gridW(row + 2, vector<_8SSEDT_struct>(col + 2, {infinity, infinity}));
    _8SSEDT_struct **gridB = new _8SSEDT_struct*[row + 2], **gridW = new _8SSEDT_struct*[row + 2];
        
    #pragma omp parallel for
    for (int i = 0; i < row + 2; i++)
    {
        gridB[i] = new _8SSEDT_struct[col + 2];
        gridW[i] = new _8SSEDT_struct[col + 2];
                
        #pragma omp parallel for
        for (int j = 0; j < col + 2; j++)
            gridB[i][j] = {0, 0}, gridW[i][j] = {infinity, infinity};
    }

    #pragma omp parallel for
    for (int i = 0; i <= row + 1; ++i)
        gridB[i][0] = gridB[i][col + 1] = {infinity, infinity};
    
    #pragma omp parallel for
    for (int i = 0; i <= col + 1; ++i)
        gridB[0][i] = gridB[row + 1][i] = {infinity, infinity};

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            if (img.at<uchar>(i, j) == 0)
                gridB[i + 1][j + 1] = {infinity, infinity}, gridW[i + 1][j + 1] = {0, 0};
    
    for (int i = 1; i <= row; ++i)
    {
        for (int j = 1; j <= col; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {                
                int x = i + offsetX[k], y = j + offsetY[k];

                gridB[i][j].toMinVal(gridB[x][y] + _8SSEDT_struct{offsetX[k], offsetY[k]});
                gridW[i][j].toMinVal(gridW[x][y] + _8SSEDT_struct{offsetX[k], offsetY[k]});
            }
        }

        for (int j = col; j > 0; --j)
        {
            gridB[i][j].toMinVal(gridB[i][j + 1] + _8SSEDT_struct{0, 1});
            gridW[i][j].toMinVal(gridW[i][j + 1] + _8SSEDT_struct{0, 1});
        }
    }

    for (int i = row; i > 0; --i)
    {
        for (int j = col; j > 0; --j)
        {            
            for (int k = 4; k < 8; ++k)
            {
                int x = i + offsetX[k], y = j + offsetY[k];
                gridB[i][j].toMinVal(gridB[x][y] + _8SSEDT_struct{offsetX[k], offsetY[k]});
                gridW[i][j].toMinVal(gridW[x][y] + _8SSEDT_struct{offsetX[k], offsetY[k]});
            }
        }

        for (int j = 1; j <= col; ++j)
        {
            gridB[i][j].toMinVal(gridB[i][j - 1] + _8SSEDT_struct{0, -1});
            gridW[i][j].toMinVal(gridW[i][j - 1] + _8SSEDT_struct{0, -1});
        }
    }

    cv::Mat _img(row, col, CV_64F, cv::Scalar(0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            _img.at<double>(i, j) = sqrt(gridB[i + 1][j + 1].getDistance()) - sqrt(gridW[i + 1][j + 1].getDistance());

    #pragma omp parallel for
    for (int i = 0; i < row + 2; i++)
        delete[] gridB[i], delete[] gridW[i];
    delete[] gridB, delete[] gridW;

    return _img;
}

cv::Mat blendSDF(const cv::Mat imgs[], int size)
{
    if (size <= 1)
        throw std::invalid_argument("size must be greater than 1");
    
    int row = imgs[0].rows, col = imgs[0].cols;
    double scale = 1.0 / (size - 1);
    cv::Mat res = cv::Mat(row, col, CV_8UC1, cv::Scalar(0));
    
    for (int n = 0; n < size - 1; ++n)
    {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
            {
                double a = imgs[n].at<double>(i, j),
                    b = imgs[n + 1].at<double>(i, j);
                double t = std::abs(a) / (std::abs(a) + std::abs(b));
                if (a * b < 0)
                    res.at<uchar>(i, j) = (uchar)((1 - (n + t) * scale) * 255.0);
                else if (n == 0 && a < 0)
                    res.at<uchar>(i, j) = 255;
            }
    }
    return res;
}

cv::Mat blendSDF(const vector<Mat>& imgs)
{
    int size = imgs.size();
    if (size <= 1)
        throw std::invalid_argument("size must be greater than 1");
    
    int row = imgs[0].rows, col = imgs[0].cols;
    double scale = 1.0 / (size - 1);
    cv::Mat res = cv::Mat(row, col, CV_8UC1, cv::Scalar(0));
    
    for (int n = 0; n < size - 1; ++n)
    {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
            {
                double a = imgs[n].at<double>(i, j),
                    b = imgs[n + 1].at<double>(i, j);
                double t = std::abs(a) / (std::abs(a) + std::abs(b));
                if (a * b < 0)
                    res.at<uchar>(i, j) = (uchar)((1 - (n + t) * scale) * 255.0);
                else if (n == 0 && a < 0)
                    res.at<uchar>(i, j) = 255;
            }
    }
    return res;
}

#endif // SDF_HPP_INCLUDED