#ifndef _MYSQL_HPP_
#define _MYSQL_HPP_

#include <stdlib.h>
#include <sstream>
#include <stdexcept>

#include "mysql_connection.h"

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

#include "common.hpp"
#include "yolo.hpp"
#include "config.hpp"

#define MYSQL_HOST "127.0.0.1"
#define MYSQL_USER "root"
#define MYSQL_PASSWD "1AlgorithM"
#define MYSQL_DB "meter_readings"

class mysqlServer 
{
    private:
        std::string url;
        const std::string user;
        const std::string passwd;
        const std::string database;
        int debug_on;

        sql::Driver* driver;
        std::auto_ptr<sql::Connection> con;
        std::auto_ptr<sql::Statement> stmt;

    public:
        mysqlServer(std::string sql_url, std::string user, std::string passwd, std::string database, int debug_on);
        ~mysqlServer();

        void init_camera_instruments(std::vector<FrameInfo> &frame_batch, std::vector<std::string> &urls); // initialize the Instruments table in the database
        void insert_readings(std::vector<MeterInfo> &meters); // insert readings into the Readings table in the database
};

#endif