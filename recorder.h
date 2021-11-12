#ifndef RECORDER_H
#define RECORDER_H

#include <vector>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <Eigen/Dense>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

namespace rabit{

class uuid_recorder {
    private:
    
    const std::string path;
    HighFive::File    file;
    
    const unsigned long n_rows;
    unsigned long row_idx;

    std::vector<Eigen::MatrixXd*>  observed_matrices;
    std::vector<HighFive::DataSet> matrix_tables;

    std::vector<Eigen::VectorXd*>  observed_vectors;
    std::vector<HighFive::DataSet> vector_tables;


    public:

    uuid_recorder(const std::string &directory, const unsigned long n_rows):
        path(directory + boost::uuids::to_string(boost::uuids::random_generator()()) + ".h5"),
        file(path,  HighFive::File::ReadWrite |
                    HighFive::File::Create |  
                    HighFive::File::Truncate),
        n_rows(n_rows),
        row_idx(0) {}

    std::string get_filename()
    {
        return path;
    }

    void create_table(std::string name, Eigen::MatrixXd *m)
    {
        observed_matrices.push_back(m);
        unsigned long n_cols = static_cast<unsigned long>(m->rows() * m->cols());
        HighFive::DataSet table = (file.createDataSet<double>(name, 
                                HighFive::DataSpace{n_rows, n_cols}));
        matrix_tables.push_back(table);
    }

    void create_table(std::string name, Eigen::VectorXd *m)
    {
        observed_vectors.push_back(m);
        unsigned long n_cols = static_cast<unsigned long>(m->rows() * m->cols());
        HighFive::DataSet table = (file.createDataSet<double>(name, 
                                HighFive::DataSpace{n_rows, n_cols}));
        vector_tables.push_back(table);
    }

    void record()
    {
        if(row_idx >= n_rows) {
            std::cerr << "Recorder file " + path + " is out of space\n";
            return;
        }

        for(unsigned int i = 0; i < observed_matrices.size(); i++) {
            Eigen::MatrixXd *m = observed_matrices[i];
            std::vector<double> m_std_vector;
            HighFive::DataSet &s = matrix_tables[i];

            for(int a = 0; a < m->rows(); a++) {
                for(int b = 0; b < m->cols(); b++)
                    m_std_vector.push_back((*m)(a,b));
            }

            unsigned long n_cols = static_cast<unsigned long>(m->rows() * m->cols());            
            s.select({row_idx, 0}, {1, n_cols}).write(m_std_vector);
        }

        for(unsigned int i = 0; i < observed_vectors.size(); i++) {
            Eigen::VectorXd *m = observed_vectors[i];
            std::vector<double> m_std_vector;
            HighFive::DataSet &s = vector_tables[i];

            for(int a = 0; a < m->rows(); a++) {
                for(int b = 0; b < m->cols(); b++)
                    m_std_vector.push_back((*m)(a,b));
            }

            unsigned long n_cols = static_cast<unsigned long>(m->rows() * m->cols());            
            s.select({row_idx, 0}, {1, n_cols}).write(m_std_vector);
        }

        row_idx++;
    }
};

}

#endif
