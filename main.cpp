// extrapolation to real chemical potentials for lattice QCD simulations
// using previously determined sector coefficients

// used headers and libraries
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>

// ---------------------------------------------------------------------------------------

// constants
int const maxIteration = 10000;
int const muBNum = 200;
double const eps = 1e-10;
double const muBMin = 0.01;
double const muBMax = 2.00;

// ---------------------------------------------------------------------------------------

// read file with given file name
// read file with dataset into a raw matrix
auto ReadFile = [](std::string const &fileName) {
    // start reading
    std::ifstream fileToRead;
    fileToRead.open(fileName);

    // determine number of columns
    std::string firstLine;
    std::getline(fileToRead, firstLine);
    std::stringstream firstLineStream(firstLine);

    // number of columns in given file
    int numOfCols = 0;
    std::string tmpString;
    // count number of writes to a temporary string container
    while (firstLineStream >> tmpString)
    {
        numOfCols++;
    }
    fileToRead.close();

    // string for all the lines
    std::string line;

    // data structure (raw matrix) to store the data
    Eigen::MatrixXd rawDataMat(0, numOfCols);

    // reopen file
    fileToRead.open(fileName);
    // check if open
    if (fileToRead.is_open())
    {
        // read line by line
        int i = 0;
        while (std::getline(fileToRead, line))
        {
            // using stringstream to write matrix
            std::stringstream dataStream(line);
            rawDataMat.conservativeResize(i + 1, numOfCols);
            for (int j = 0; j < numOfCols; j++)
            {
                dataStream >> rawDataMat(i, j);
            }
            i++;
        }
        // close file
        fileToRead.close();
    }
    // error check
    else
    {
        std::cout << "ERROR\nProblem occured while reading given file." << std::endl;
        std::exit(-1);
    }

    // return raw data matrix
    return (Eigen::MatrixXd)rawDataMat;
};

// ---------------------------------------------------------------------------------------

// basis functions for susceptibilities (real chemical potential)
auto BasisFuncReal = [](int const &B, int const &S, int const &BOrder, int const &SOrder, double const &muB, double const &muS) {
    // total number of partial derivations
    int FullOrder = BOrder + SOrder;
    // prefactor
    double preFactor = std::pow(B, BOrder) * std::pow(-S, SOrder);
    // odd derivative
    if (FullOrder % 4 == 1 || FullOrder % 4 == 3)
    {
        return preFactor * std::sinh(B * muB - S * muS);
    }
    // even derivative
    else if (FullOrder % 4 == 2 || FullOrder % 4 == 0)
    {
        return preFactor * std::cosh(B * muB - S * muS);
    }
    else
    {
        std::cout << "ERROR\nInvalid derivative order." << std::endl;
        std::exit(-1);
    }
};

// ---------------------------------------------------------------------------------------

// ZS function (real chemical potential)
auto ZFuncReal = [](double const &muB, double const &muS, int const &BOrder, int const &SOrder, std::vector<std::pair<int, int>> const &SectorNumbers, Eigen::VectorXd const &SectorCoeffs) {
    // number of analysed sectors
    int N = static_cast<int>(SectorNumbers.size());
    // container for results
    double res = 0.;
    // accumulating results
    for (int i = 0; i < N; i++)
    {
        // baryon number
        int B = SectorNumbers[i].first;
        // strangeness
        int S = SectorNumbers[i].second;
        // update result
        res += SectorCoeffs[i] * BasisFuncReal(B, S, BOrder, SOrder, muB, muS);
    }
    // return result
    return res;
};

// ---------------------------------------------------------------------------------------

// jackknife error estimation
// calculate variance (for jackknife samples)
auto JCKVariance = [](Eigen::VectorXd const &JCKSamples) {
    // size of vector
    int N = JCKSamples.size();
    // pre-factor
    double preFactor = (double)(N - 1) / N;
    // estimator / mean
    double estimator = JCKSamples.mean();
    // calculate variance
    double var = 0.;
    for (int i = 0; i < N; i++)
    {
        double val = JCKSamples(i) - estimator;
        var += val * val;
    }
    // return variance
    return preFactor * var;
};

// main function
// argv[1] --> file name to use
// argv[2] --> where to cut sectors (B = 2 or 3)
int main(int argc, char **argv)
{
    // check length of argument list
    int const argcExpected = 3;
    if (argc > argcExpected)
    {
        std::cout << "ERROR\nNot enough arguments: " << argc << " is given instead of " << argcExpected << "." << std::endl;
        std::exit(-1);
    }

    // reading argument list
    // sector coefficients at fixed temperature
    std::string const fileName = argv[1];
    // where to cut sector (B = 2 or 3)
    int const SectorCut = std::atoi(argv[2]);
    if (SectorCut != 2 && SectorCut != 3)
    {
        std::cout << "ERROR\nSector cut parameter is not appropriate: instead of " << SectorCut << " it should be 2 or 3" << std::endl;
        std::exit(-2);
    }

    // raw data matrix: sector coefficients and its error
    Eigen::MatrixXd PMatrix = ReadFile(fileName);
    // number of data points
    Eigen::VectorXd muB(muBNum);
    for (int i = 0; i < muBNum; i++)
    {
        muB(i) = muBMin + i * muBMax / muBNum;
    }
    Eigen::MatrixXd muS = Eigen::MatrixXd::Zero(muBNum, PMatrix.cols());

    // used sectors
    std::vector<std::pair<int, int>> SectorNumbers;
    if (SectorCut == 2)
        SectorNumbers = {{1, 0}, {0, 1}, {1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {0, 2}, {0, 3}};
    else
        SectorNumbers = {{1, 0}, {0, 1}, {1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {0, 2}, {0, 3}, {3, 0}, {3, 1}, {3, 2}, {3, 3}};

    // determine muS(muB) dependence via making the expactation value of strangeness to be zero
    // using Newton iteration
    for (int iMuB = 0; iMuB < muBNum; iMuB++)
    {
        for (int iSample = 0; iSample < PMatrix.cols(); iSample++)
        {
            // Newton iteration to find roots
            // 0th step
            double guess = 1.;
            // 1st step
            double res = guess - ZFuncReal(muB(iMuB), guess, 0, 1, SectorNumbers, PMatrix.col(iSample)) / ZFuncReal(muB(iMuB), guess, 0, 2, SectorNumbers, PMatrix.col(iSample));
            // start iteration
            int nIter = 1;
            while (std::abs(guess - res) > eps && nIter < maxIteration)
            {
                guess = res;
                res = guess - ZFuncReal(muB(iMuB), guess, 0, 1, SectorNumbers, PMatrix.col(iSample)) / ZFuncReal(muB(iMuB), guess, 0, 2, SectorNumbers, PMatrix.col(iSample));
                nIter++;
            }
            muS(iMuB, iSample) = res;
        }
    }

    // susceptibility ratios and their estimated errors
    Eigen::VectorXd Z12 = Eigen::VectorXd::Zero(muBNum);
    Eigen::VectorXd Z31 = Eigen::VectorXd::Zero(muBNum);
    Eigen::VectorXd Z42 = Eigen::VectorXd::Zero(muBNum);
    Eigen::MatrixXd Z12Err = Eigen::MatrixXd::Zero(muBNum, PMatrix.cols() - 1);
    Eigen::MatrixXd Z31Err = Eigen::MatrixXd::Zero(muBNum, PMatrix.cols() - 1);
    Eigen::MatrixXd Z42Err = Eigen::MatrixXd::Zero(muBNum, PMatrix.cols() - 1);

    // extra data 1
    Eigen::VectorXd Diff31 = Eigen::VectorXd::Zero(muBNum);
    Eigen::VectorXd Diff42 = Eigen::VectorXd::Zero(muBNum);
    Eigen::MatrixXd Diff31Err = Eigen::MatrixXd::Zero(muBNum, PMatrix.cols() - 1);
    Eigen::MatrixXd Diff42Err = Eigen::MatrixXd::Zero(muBNum, PMatrix.cols() - 1);
    for (int i = 0; i < muBNum; i++)
    {
        // results
        Z12(i) = ZFuncReal(muB(i), muS(i, 0), 1, 0, SectorNumbers, PMatrix.col(0)) / ZFuncReal(muB(i), muS(i, 0), 2, 0, SectorNumbers, PMatrix.col(0));
        Z31(i) = ZFuncReal(muB(i), muS(i, 0), 3, 0, SectorNumbers, PMatrix.col(0)) / ZFuncReal(muB(i), muS(i, 0), 1, 0, SectorNumbers, PMatrix.col(0));
        Z42(i) = ZFuncReal(muB(i), muS(i, 0), 4, 0, SectorNumbers, PMatrix.col(0)) / ZFuncReal(muB(i), muS(i, 0), 2, 0, SectorNumbers, PMatrix.col(0));

        // extra 1 results
        Diff31(i) = Z31(i) - Z31(0);
        Diff42(i) = (Z42(i) - Z42(0)) / 3.;

        // errors from jackknife samples
        for (int iSample = 0; iSample < PMatrix.cols() - 1; iSample++)
        {
            Z12Err(i, iSample) = ZFuncReal(muB(i), muS(i, iSample + 1), 1, 0, SectorNumbers, PMatrix.col(iSample + 1)) / ZFuncReal(muB(i), muS(i, iSample + 1), 2, 0, SectorNumbers, PMatrix.col(iSample + 1));
            Z31Err(i, iSample) = ZFuncReal(muB(i), muS(i, iSample + 1), 3, 0, SectorNumbers, PMatrix.col(iSample + 1)) / ZFuncReal(muB(i), muS(i, iSample + 1), 1, 0, SectorNumbers, PMatrix.col(iSample + 1));
            Z42Err(i, iSample) = ZFuncReal(muB(i), muS(i, iSample + 1), 4, 0, SectorNumbers, PMatrix.col(iSample + 1)) / ZFuncReal(muB(i), muS(i, iSample + 1), 2, 0, SectorNumbers, PMatrix.col(iSample + 1));

            // extra 1 results
            Diff31Err(i, iSample) = Z31Err(i, iSample) - Z31Err(0, iSample);
            Diff42Err(i, iSample) = (Z42Err(i, iSample) - Z42Err(0, iSample)) / 3.;
        }

        // write to screen
        std::cout << muB(i) << " "
                  //<< muS.col(0)(i) << " " << std::sqrt(JCKVariance(muS.row(i).segment(1, PMatrix.cols() - 1))) << " "
                  //<< Z12(i) << " " << std::sqrt(JCKVariance(Z12Err.row(i))) << " "
                  //<< Z31(i) << " " << std::sqrt(JCKVariance(Z31Err.row(i))) << " "
                  //<< Z42(i) << " " << std::sqrt(JCKVariance(Z42Err.row(i))) << std::endl;
                  << Diff31(i) << " " << std::sqrt(JCKVariance(Z31Err.row(i))) << " "
                  << Diff42(i) << " " << std::sqrt(JCKVariance(Z42Err.row(i))) << std::endl;
    }
}