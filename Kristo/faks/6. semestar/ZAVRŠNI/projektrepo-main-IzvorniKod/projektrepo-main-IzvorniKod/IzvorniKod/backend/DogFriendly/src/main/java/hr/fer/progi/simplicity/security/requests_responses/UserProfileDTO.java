package hr.fer.progi.simplicity.security.requests_responses;


import java.util.List;

public class UserProfileDTO {

    private String username;
    private String email;
    private List<UserRatingDTO> ratings;

    public UserProfileDTO(){
    }

    public UserProfileDTO(String username, String email, List<UserRatingDTO> ratings) {
        this.username = username;
        this.email = email;
        this.ratings = ratings;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public List<UserRatingDTO> getRatings() {
        return ratings;
    }

    public void setRatings(List<UserRatingDTO> ratings) {
        this.ratings = ratings;
    }
}
