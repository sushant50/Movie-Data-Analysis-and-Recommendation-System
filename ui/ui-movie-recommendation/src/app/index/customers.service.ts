import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';


@Injectable({
    providedIn: 'root'
  })
  
  export class CustomersService {

    constructor(
        private http: HttpClient) {

    }

getData(url) {
   return this.http.get(url)
}

}