#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
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

cv::Mat color2bit()
{
    const cv::Mat srcImage = cv::imread(INPUT_IMAGE, CV_8UC4);
    cv::imshow("colorful kunkun", srcImage);
    cv::Mat dstImage(srcImage.size(), CV_8UC1);
    constexpr auto cv8uc1 = CV_8UC1;
    auto type = dstImage.type();
    cv::threshold(srcImage, dstImage, 254, 255, cv::THRESH_BINARY);
    cv::imshow("bin kunkun", dstImage);
    return dstImage;
}

std::vector<cv::Point2f> getPoints(const cv::Mat& binImage)
{
    std::vector<cv::Point2f> points;
    //extracting the contours from the given binary image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat srcImage(binImage.size(), CV_8UC1, cv::Scalar(0, 0, 0));
    cv::cvtColor(binImage, srcImage, cv::COLOR_RGB2GRAY);
    cv::findContours(srcImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    std::cout << "Total Number of Contours found =" << contours.size() << std::endl;
    //create new image to draw contours
    cv::Mat contourImage(srcImage.size(), CV_8UC1, cv::Scalar(0, 0, 0));
    //draw contours
    cv::drawContours(contourImage, contours, -1, cv::Scalar(255, 255, 255), 2);
    cv::imshow("Contours", contourImage);
    points.reserve(contours[1].size());
    auto& kunContour = contours[1];
    for (const auto& p : contours[1])
    {
        points.push_back(p);
    }
    std::reverse(points.begin(), points.end());

    cv::Mat newImage(binImage.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    // draw colored contour red to blue for contour direction
    for (int i = 0; i < points.size(); ++i)
    {
        float phi = (float)i / points.size();
        cv::circle(newImage, points[i], 0, cv::Scalar(255.0 * (1.0f - phi), 0, 255.0 * phi));
    }
    cv::imshow("colored contour kunkun", newImage);
    return points;
}

int findNearestPointIdx(const std::vector<cv::Point2f>& boundary, const cv::Point2f& pt)
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

void prepareSpecimen(const std::vector<cv::Point2f>& boundary)
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
            const int min_idx = findNearestPointIdx(boundary, cv::Point2f(pos_x, pos_y));
            
            cv::Point2f B1, B2;
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

            const Eigen::Vector2d eB1(B1.x, B1.y), eB2(B1.x, B2.y);
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
    
    std::ofstream outputFile(OUTPUT_P4P, std::ios::out);
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
    cv::Mat binImage= color2bit();
    std::vector<cv::Point2f> ptList = getPoints(binImage);
    prepareSpecimen(ptList);
    cv::waitKey(0);
    return 0;
}
