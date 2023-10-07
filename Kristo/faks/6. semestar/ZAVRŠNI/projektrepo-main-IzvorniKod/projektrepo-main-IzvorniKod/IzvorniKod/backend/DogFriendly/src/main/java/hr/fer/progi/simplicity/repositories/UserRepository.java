package hr.fer.progi.simplicity.repositories;

import hr.fer.progi.simplicity.entities.User;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<User, Long> {
    User getUserById(Long id);
    User findByUsername(String username);
    User findByEmail(String email);
    void deleteByUsername(String username);

}
