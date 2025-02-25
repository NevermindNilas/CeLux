#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Haldclutsrc : public FilterBase {
public:
    /**
     * Provide an identity Hald CLUT.
     */
    /**
     * set level
     * Type: Integer
     * Required: No
     * Default: 6
     */
    void setLevel(int value);
    int getLevel() const;

    /**
     * set video rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    /**
     * set video duration
     * Aliases: d
     * Type: Duration
     * Required: No
     * Default: -1
     */
    void setDuration(int64_t value);
    int64_t getDuration() const;

    /**
     * set video sample aspect ratio
     * Type: Rational
     * Required: No
     * Default: 0
     */
    void setSar(const std::pair<int, int>& value);
    std::pair<int, int> getSar() const;

    Haldclutsrc(int level = 6, std::pair<int, int> rate = std::make_pair<int, int>(0, 1), int64_t duration = 0, std::pair<int, int> sar = std::make_pair<int, int>(0, 1));
    virtual ~Haldclutsrc();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int level_;
    std::pair<int, int> rate_;
    int64_t duration_;
    std::pair<int, int> sar_;
};
