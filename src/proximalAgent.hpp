#ifndef PROXIMAL_AGENT_HPP
#define PROXIMAL_AGENT_HPP


namespace stdmMf {


class ProximalAgent : public Agent, public RngClass {

public:
    ProximalAgent(const std::shared_ptr<const Network> & network);

};


} // namespace stdmMf


#endif // PROXIMAL_AGENT_HPP
