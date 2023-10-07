package hr.fer.progi.simplicity.entities;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

@Entity (name = "webusers")
@Inheritance(strategy = InheritanceType.JOINED)
public class User {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true)
    private String username;

    @Column(unique = true)
    private String email;

    @NotNull
    private String password;

    private RoleType role;

    private boolean accountActivatedByEmail;

    public User(String username, String email, String password, RoleType role) {
        this.username = username;
        this.email = email;
        this.password = password;
        this.role = role;
        this.accountActivatedByEmail = false;
    }

    public User(){}

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
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

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public RoleType getRole() {
        return role;
    }

    public void setRole(RoleType role) {
        this.role = role;
    }

    public boolean isAccountActivatedByEmail() {
        return accountActivatedByEmail;
    }

    public void setAccountActivatedByEmail(boolean accountActivatedByEmail) {
        this.accountActivatedByEmail = accountActivatedByEmail;
    }

}
