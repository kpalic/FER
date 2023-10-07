package hr.fer.progi.simplicity.entities;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

@Entity
public class Location {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Double longitude;
    private Double latitude;

    @Column(unique = true)
    @NotNull
    private String name;

    private LocationType type;

    private Integer votesSum;
    private Integer positiveVotes;
    private Double rating;

    public Location() {

    }

    public Location(Double longitude, Double latitude, String name, LocationType type) {
        this.longitude = longitude;
        this.latitude = latitude;
        this.name = name;
        this.type = type;
        this.votesSum=0;
        this.positiveVotes=0;
        this.rating=0d;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Double getLongitude() {
        return longitude;
    }

    public void setLongitude(Double longitude) {
        this.longitude = longitude;
    }

    public Double getLatitude() {
        return latitude;
    }

    public void setLatitude(Double latitude) {
        this.latitude = latitude;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public LocationType getType() {
        return type;
    }

    public void setType(LocationType type) {
        this.type = type;
    }

    public Integer getVotesSum() {
        return votesSum;
    }

    public void setVotesSum(Integer votesSum) {
        this.votesSum = votesSum;
    }

    public Integer getPositiveVotes() {
        return positiveVotes;
    }

    public void setPositiveVotes(Integer positiveVotes) {
        this.positiveVotes = positiveVotes;
    }

    public Double getRating() {
        return rating;
    }

    public void setRating(Double rating) {
        this.rating = rating;
    }
}
