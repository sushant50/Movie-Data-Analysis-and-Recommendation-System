
import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CustomersService } from './customers.service';
import { ActivatedRoute, Params, Router } from '@angular/router';

@Component({
    selector: 'app-index',
    templateUrl: './index.component.html',
    styleUrls: ['./index.component.scss']
})
export class IndexComponent implements OnInit {
  movieName: any;
  constructor(private router:Router,private customersservice: CustomersService, public activatedRoute: ActivatedRoute) {
        
  }
  movieArray = []
  movies = []
  ngOnInit(): void 
  {
    
    
    this.activatedRoute.queryParams.subscribe((params: Params) => {
      this.movieName = params['recommendBy'] || '';
      if(this.movieName) {
        this.getClickedMovieData(this.recommender_api + "?movie="+this.movieName);

      }
      else {
        this.getLandingData(this.recommender_api);

      }
    })
  }

  title = 'my-project';
  api_key = 'https://api.themoviedb.org/3/search/movie?api_key=15d2ea6d0dc1d476efbca3eba2b9bbfb'
  poster_api='http://image.tmdb.org/t/p/w500//';
  recommender_api = 'http://127.0.0.1:5002/recommender'

  
  public getLandingData(url) {
    this.customersservice.getData(url)
      .subscribe((res) => {
        if(typeof res == "string") {
          let a;
          let b = [];
          a = Object.keys(JSON.parse(<any>res))
          for(let i in a) {
            let obj = {name:'', api_name:''}
            obj.name =  a[i].slice(0, -7)
            obj.api_name = a[i]
            b.push(obj)
          }
          for(let i in b) {
            let url = this.api_key + "&query=" + b[i].name
            let name = b[i].api_name
            this.getCompleteMovieData(url, name)

          }
        }
    else {
      console.log(res)
    }
    })
    
}
public getClickedMovieData(url) {
  this.customersservice.getData(url)
    .subscribe((res) => {
      let a;
      let b = [];
      a = JSON.parse(<any>res)
      for(let i in a) {
        let obj = {name:'', api_name:''}
        obj.name =  a[i].slice(0, -7)
        obj.api_name = a[i]
        b.push(obj)
      }
      for(let i in b) {
        let url = this.api_key + "&query=" + b[i].name
        let name = b[i].api_name
        this.getCompleteMovieData(url, name)

      }
  })
  
}

public getCompleteMovieData(url, name){
  this.movieArray = []
  this.customersservice.getData(url)
  .subscribe((res) => {
    let a;
    a = <any>res;
    if(a.results[0]){
      a.results[0].poster_path = this.poster_api +  a.results[0].poster_path
      a.results[0].api_name = name
      this.movieArray.push(a.results[0])
    }

  })
}

callApi(name) {
  if(name){
    this.router.navigate(['/'], { queryParams: {
      recommendBy: name
    } });
  }
  else {
    this.router.navigate(['/'], { queryParams: {
    } });
  }

}

}
