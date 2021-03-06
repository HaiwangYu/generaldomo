#ifndef GENERALDOMO_WORKER_HPP_SEEN
#define GENERALDOMO_WORKER_HPP_SEEN

#include "generaldomo/util.hpp"
#include "generaldomo/logging.hpp"

namespace generaldomo {

    /*! The generaldomo worker API
     *
     * Applications may use a Worker to simplify participating in the
     * GDP worker subprotocol.  The protocol is exercised through
     * calls to Worker::work().
     *
     * The application must arrange to use the "clientish" socket
     * corresponding to what is in use by the broker but otherwise,
     * differences related to the socket type are subsequently erased
     * by this class.
     */

    class Worker {
    public:
        /// Create a worker providing service.  Caller keeps socket eg
        /// so to poll it along with others.
        Worker(zmq::socket_t& sock, std::string broker_address,
               std::string service, logbase_t& log);
        ~Worker();

        // API methods

        /// Send reply from last request, if any, and get new request.
        /// Both reply and request multiparts begin with "Frames 5+
        /// request body" of 7/MDP.  This will only return when there
        /// is a request.  It will internally allow a timeout and a
        /// reconnect to the broker.  If owner of the Worker needs to
        /// get time to do other things while waiting for a request
        /// then send() and recv() may be used.
        zmq::multipart_t work(zmq::multipart_t& reply);

        /// Recieve a request.  Note, if no request is pending, this
        /// will wait for at most one heartbeat and return leaving the
        /// request empty.  If the request is not empty a subsequent
        /// send() shall be made.        
        void recv(zmq::multipart_t& request);
        /// Send a reply.  A reply must only be sent in response to a
        /// request.  Note, unlike using work() it is not required,
        /// but still allowed, to send an initial empty reply.
        /// Empties will simply be ignored.
        void send(zmq::multipart_t& reply);


    private:
        zmq::socket_t& m_sock;
        std::string m_address;
        std::string m_service;
        logbase_t& m_log;
        int m_liveness{HEARTBEAT_LIVENESS};
        time_unit_t m_heartbeat{HEARTBEAT_INTERVAL};
        time_unit_t m_reconnect{HEARTBEAT_INTERVAL};
        time_unit_t m_heartbeat_at{0};
        bool m_expect_reply{false};
        std::string m_reply_to{""};

    private:

        std::function<void(zmq::socket_t& server_socket,
                           zmq::multipart_t& mmsg)> really_recv;
        std::function<void(zmq::socket_t& server_socket,
                           zmq::multipart_t& mmsg)> really_send;

        void connect_to_broker(bool reconnect = true);

    };


    // An echo worker actor function
    void echo_worker(zmq::socket_t& pipe, std::string address, int socktype);

}

#endif
