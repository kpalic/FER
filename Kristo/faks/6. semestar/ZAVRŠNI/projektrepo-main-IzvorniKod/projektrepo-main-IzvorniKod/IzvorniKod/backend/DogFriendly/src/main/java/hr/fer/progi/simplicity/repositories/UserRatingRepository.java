package hr.fer.progi.simplicity.repositories;

import hr.fer.progi.simplicity.entities.Location;
import hr.fer.progi.simplicity.entities.User;
import hr.fer.progi.simplicity.entities.UserRating;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface UserRatingRepository extends JpaRepository<UserRating, Long> {
   List<UserRating> findAllByUser(User user);
   List<UserRating> findAllByLocation(Location location);
   UserRating getUserRatingById(Long id);

}
