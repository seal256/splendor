#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <string>

#include "json.hpp"

#include "game_state.h"

namespace splendor {
   
constexpr int NUM_GEMS = 6;
constexpr int CARD_LEVELS = 3;

extern const std::vector<std::string> ACTIONS_STR;

struct GemSet {
    std::array<int, NUM_GEMS> gems;

    GemSet() {gems.fill(0);};
    static GemSet from_str(const std::string& input);
    int sum() const;
    int unique() const;
};

std::ostream& operator<<(std::ostream& os, const GemSet& gem_set);
// std::istream& operator>>(std::istream& is, GemSet& gem_set);

struct Noble {
    int points; // Number of win points
    GemSet price;

    Noble(int points = 0, const GemSet& price = GemSet()) : points(points), price(price) {};
    static Noble from_str(const std::string& input); 
    std::string to_str() const;
};

std::ostream& operator<<(std::ostream& os, const Noble& noble);

struct Card {
    int gem = 0;    // Title gem
    int points = 0; // Number of win points
    GemSet price;

    Card(int gem = 0, int points = 0, const GemSet& price = GemSet()) : gem(gem), points(points), price(price) {};
    static Card from_str(const std::string& input); 
    std::string to_str() const;
};

std::ostream& operator<<(std::ostream& os, const Card& card);

extern const std::vector<Noble> NOBLES;
extern const std::vector<std::vector<Card>> CARDS;

struct SplendorPlayerState {
    int id;                         // player's index
    GemSet card_gems;               // Gems from acquired cards
    GemSet gems;                    // Gems on the table
    std::vector<const Card *> hand_cards;   // Cards in hand TODO: max size of this array is 3!
    int points = 0;                 // Winning points from all acquired cards
};

std::ostream& operator<<(std::ostream& os, const SplendorPlayerState& player_state);

enum class ActionType {
    SKIP,           // skip the move
    TAKE,           // take gems
    RESERVE,        // reserve card
    PURCHASE,       // purchase table card
    PURCHASE_HAND,  // purchase hand card
    NEW_TABLE_CARD  // new table card from deck
};
 
struct Action {
    ActionType type;
    GemSet gems; // gems to take
    int level = -1;   // card level (for reserve, purchase, new_table_card)
    int pos = -1;     // postion of the card to take

    Action(ActionType type, GemSet gems, int level, int pos) : type(type), gems(gems), level(level), pos(pos) {};
    Action(ActionType type, GemSet gems) : type(type), gems(gems) {};
    Action(ActionType type, int level, int pos) : type(type), level(level), pos(pos) {};
    Action(ActionType type, int pos) : type(type), pos(pos) {};
    Action(ActionType type) : type(type) {};
    Action() : type(ActionType::SKIP) {};
    static Action from_str(const std::string& input);
    std::string to_str() const;
};

std::ostream& operator<<(std::ostream& os, const Action& action);

struct SplendorGameRules {
    int num_players = 4;
    int max_open_cards = 4;        // Open cards on table
    int max_hand_cards = 3;        // Max cards in player hand
    int win_points = 15;           // Points required to win
    int max_player_gems = 10;      // Max gems a player can hold
    int max_nobles = 5;            // Max nobles (depends on num_players)
    int max_gems_take = 3;         // Max gems to take
    int max_same_gems_take = 2;    // Max same-color gems to take
    int min_same_gems_stack = 4;   // Min gem stack size to take 2 same-color gems
    int max_gold = 5;              // Max gold gems
    int max_gems = 7;              // Max same-color gems on table (except gold)

    SplendorGameRules(int players) : num_players(players) {
        max_nobles = num_players + 1;
        max_gems = (num_players < 4) ? 2 + num_players : 7;
    }
};

extern const std::unordered_map<int, SplendorGameRules> DEFAULT_RULES;

enum class ActionError {
    NONE,
    INVALID_ACTION_ID,
    CANNOT_TAKE_MORE_THAN_MAX_GEMS,
    NOT_ENOUGH_GEMS_IN_STACK,
    CANNOT_TAKE_MORE_THAN_MAX_IDENTICAL_GEMS,
    MUST_TAKE_ALL_IDENTICAL_OR_ALL_DIFFERENT,
    PLAYER_CANNOT_HAVE_MORE_GEMS,
    CANNOT_TAKE_GOLD_GEM,
    NOT_ENOUGH_GEMS_ON_TABLE,
    INVALID_DECK_LEVEL,
    INVALID_CARD_POSITION,
    PLAYER_CANNOT_RESERVE_MORE_CARDS,
    PLAYER_CANNOT_AFFORD_CARD,
    GAME_REQUIRES_NEW_TABLE_CARD,
    GAME_DOES_NOT_NEED_NEW_TABLE_CARD,
    INVALID_ACTION_TYPE
};

class SplendorGameState : public GameState {
public:
    const SplendorGameRules* rules;
    int round;
    int player_to_move;
    int skips;                 // number of players that skipped move in this round. If all players skipped, the game ends
    bool table_card_needed;    // indicates that previous player just purchased a table card
    int deck_level;            // the level of the deck that will be used to select card if table_card_needed is True
    std::vector<const Noble*> nobles;               
    std::vector<std::vector<const Card*>> decks;
    std::vector<std::vector<const Card*>> cards;
    GemSet gems;
    std::vector<SplendorPlayerState> players;

public:
    SplendorGameState(int num_players, const SplendorGameRules* rules = nullptr);

    std::vector<int> get_actions() const override;
    int active_player() const override;
    void apply_action(const int action) override;
    bool is_terminal() const override;
    std::vector<double> rewards() const override;
    std::shared_ptr<GameState> clone() const override;
    int move_num() const override;

    void print(std::ostream& os) const override;
    void to_json(nlohmann::json& j) const override;

private:
    std::pair<bool, ActionError> verify_action(const int action_id) const;
    void increment_player_to_move();
    void set_table_card_needed(int level);
    bool player_can_afford_card(const SplendorPlayerState& player, const Card& card) const;
    void purchase_card(SplendorPlayerState& player, const Card& card);
    void get_noble(SplendorPlayerState& player);
    std::vector<int> _get_winners() const;
};

std::ostream& operator<<(std::ostream& os, const SplendorGameState& state);


void to_json(nlohmann::json& j, const GemSet& gem_set);
void to_json(nlohmann::json& j, const SplendorPlayerState& player_state);
void to_json(nlohmann::json& j, const SplendorGameRules& rules);
void to_json(nlohmann::json& j, const SplendorGameState& state);

} // namespace splendor