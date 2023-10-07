package hr.fer.progi.simplicity.entities;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

@Entity
public class UserRating {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotNull
    @OneToOne
    @JoinColumn(name = "userId", referencedColumnName = "id")
    private User user;

    @NotNull
    @OneToOne
    @JoinColumn(name = "locationId", referencedColumnName = "id")
    private Location location;

    @NotNull
    private RatingType ratingType;

    public UserRating(){

    }

    public UserRating(User user, Location location, RatingType ratingType) {
        this.user = user;
        this.location = location;
        this.ratingType = ratingType;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public Location getLocation() {
        return location;
    }

    public void setLocation(Location location) {
        this.location = location;
    }

    public RatingType getRatingType() {
        return ratingType;
    }

    public void setRatingType(RatingType ratingType) {
        this.ratingType = ratingType;
    }
}
