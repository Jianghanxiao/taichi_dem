#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "Eigen/Dense"

static constexpr double sphere_z = 0.0;
static constexpr double sphere_radius = 1.0;
static constexpr double sphere_density = 1000.0;
static constexpr double sphere_volume = 4.0 / 3.0 * CV_PI * sphere_radius * sphere_radius * sphere_radius;
static constexpr double sphere_mass = sphere_density * sphere_volume;

struct Sphere
{
public:
    double x, y, z, r;
    Sphere(double _x, double _y, double _z, double _r)
    {
        x = _x; y = _y; z = _z; r = _r;
    };
};

void color2bit()
{
    const cv::Mat srcImage = cv::imread("src.png", CV_8UC4);
    cv::Mat dstImage(srcImage.size(), CV_8UC1);
    for (int i = 0; i < srcImage.rows; ++i)
        for (int j = 0; j < srcImage.cols; ++j)
        {
            const cv::Vec4b p = srcImage.at<cv::Vec4b>(i, j);
            const uchar a = p[0], b = p[1], c = p[2], d = p[3];
            if ((a == 255) && (b == 255) && (c == 255) && (d == 255)) // RGBA all 255 -> background white pixel
                dstImage.at<uchar>(i, j) = (uchar)255;
            else // else -> foreground black pixel
                dstImage.at<uchar>(i, j) = (uchar)0;
        }

    cv::Mat outImage;
    cv::resize(dstImage, outImage, dstImage.size() * 2);
    cv::imwrite("out.jpg", outImage);
}

std::vector<cv::Point2d> getPoints()
{
    const cv::Mat srcImage = cv::imread("out.jpg", CV_8UC1);
    // TODO
}

int findNearestPointIdx(const std::vector<cv::Point2d>& boundary, const cv::Point2d& pt)
{
    double min_distance = DBL_MAX;
    int min_idx = -1;
    for (int i = 0; i < boundary.size(); ++i)
    {
        const double present_distance = sqrt((boundary[i] - pt).dot(boundary[i] - pt));
        if (present_distance < min_distance)
        {
            min_idx = i;
            min_distance = present_distance;
        }
    }
    return min_idx;
}

void prepareSpecimen(const std::vector<cv::Point2d>& boundary)
{
    const cv::Rect domain = cv::boundingRect(boundary);
    std::vector<Sphere> sphList;
    // Suppose each sphere has the radius of 1.0 (px)
    for (int i = 0; i < domain.height; i += 2) // Leap twice of the sphere radius (1.0px)
    {
        const double pos_y = (double)(domain.y + i);
        for (int j = 0; j < domain.width; j += 2) // Leap twice of the sphere radius (1.0px)
        {
            const double pos_x = (double)(domain.x + j);

            // X_j = [pos_x, pos_y]
            // The nearest point A_b,i to X_j, i = min_idx
            const int min_idx = findNearestPointIdx(boundary, cv::Point2d(pos_x, pos_y));
            
            cv::Point2d B1, B2;
            if (min_idx == 0) // Consider boundary conditions
            {
                B1 = boundary[min_idx] - boundary[boundary.size() - 1];
                B2 = boundary[min_idx + 1] - boundary[min_idx];
            }
            else if (min_idx == boundary.size() - 1) // Consider boundary conditions
            {
                B1 = boundary[min_idx] - boundary[min_idx - 1];
                B2 = boundary[0] - boundary[min_idx];
            }
            else
            {
                B1 = boundary[min_idx] - boundary[min_idx - 1];
                B2 = boundary[min_idx + 1] - boundary[min_idx];
            }

            const Eigen::Vector2d eB1(B1), eB2(B2);
            Eigen::Matrix<double, 2, 2, Eigen::RowMajor> R;
            R << 0.0, 1.0,
                -1.0, 0.0;

            if ((R * eB1).transpose() * eB2 < 0)
                if ((eB1[1] * (pos_x - boundary[min_idx].x) - eB1[0] * (pos_y - boundary[min_idx].y) > 0)
                    && (eB2[1] * (pos_x - boundary[min_idx].x) - eB2[0] * (pos_y - boundary[min_idx].y) > 0))
                    // X_j in A
                    sphList.emplace_back(Sphere(pos_x, pos_y, 0.0, sphere_radius));
            else if ((R * eB1).transpose() * eB2 > 0)
                if ((eB1[1] * (pos_x - boundary[min_idx].x) - eB1[0] * (pos_y - boundary[min_idx].y) > 0)
                    || (eB2[1] * (pos_x - boundary[min_idx].x) - eB2[0] * (pos_y - boundary[min_idx].y) > 0))
                    // X_j in A
                    sphList.emplace_back(Sphere(pos_x, pos_y, sphere_z, sphere_radius));
        }
    }
    
    std::ofstream outputFile("input.p4p", std::ios::out);
    outputFile << "TIMESTEP  PARTICLES" << std::endl;
    outputFile << 0.0 << " " << sphList.size() << std::endl;
    outputFile << "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ" << std::endl;
    for (int i = 0; i < sphList.size(); ++i)
        outputFile << (i + 1) << " " << 0 << " "
        << sphere_volume << " " << sphere_mass << " "
        << sphList[i].x << " " << sphList[i].y << " " << sphList[i].z
        << " 0.0 0.0 0.0" << std::endl;
    outputFile.close();
}

int main()
{
    color2bit();
    std::vector<cv::Point2d> ptList = getPoints();
    prepareSpecimen(ptList);
    return 0;
}
